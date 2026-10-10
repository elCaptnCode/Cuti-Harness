from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.video_runtime.models import (
    ArtifactDependency,
    ArtifactInvalidationPolicy,
    MediaArtifactVersion,
    ValidationResult,
    VideoSpec,
)
from app.video_runtime.postgres_repository import PostgresVideoProjectRepository
from app.video_runtime.runtime import VideoBuildRuntime
from app.video_runtime.skill_workflows import load_workflow_skills


class _FakeExecutor:
    async def rebuild_artifact(
        self, *, build, source, completed_replacements, idempotency_key,
    ) -> MediaArtifactVersion:
        del completed_replacements
        return MediaArtifactVersion(
            project_id=build.project_id,
            type=source.type,
            uri=f"postgres-test://{source.id}",
            content_digest=idempotency_key,
        )

    async def execute_plan_step(
        self, *, build, step, completed_artifacts, idempotency_key,
        report_remote_operation=None,
    ) -> MediaArtifactVersion:
        del report_remote_operation
        metadata = {}
        if step.output_artifact_type == "timeline":
            clip = completed_artifacts[step.parameters["video_steps"][0]]
            metadata = {"timeline": {"durationSeconds": 5, "items": [{
                "artifactVersionId": clip.id, "startSeconds": 0, "durationSeconds": 5,
            }]}}
        return MediaArtifactVersion(
            project_id=build.project_id,
            type=step.output_artifact_type,
            uri=None if step.output_artifact_type in {
                "video_spec", "script", "characters", "storyboard", "timeline",
            } else f"postgres-test://{step.step_id}",
            metadata=metadata,
            content_digest=idempotency_key,
        )


class _CancellationAbsorbingExecutor(_FakeExecutor):
    """Its first step absorbs task cancellation and returns, like a broad ``except BaseException``."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.calls = 0

    async def rebuild_artifact(
        self, *, build, source, completed_replacements, idempotency_key,
    ) -> MediaArtifactVersion:
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                pass
        return await super().rebuild_artifact(
            build=build,
            source=source,
            completed_replacements=completed_replacements,
            idempotency_key=idempotency_key,
        )


@unittest.skipUnless(
    os.getenv("VIDEO_RUNTIME_DATABASE_URL"),
    "set VIDEO_RUNTIME_DATABASE_URL to run the Postgres integration test",
)
class PostgresVideoRuntimeIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_idempotent_incremental_build_survives_repository_restart(self) -> None:
        database_url = os.environ["VIDEO_RUNTIME_DATABASE_URL"]
        suffix = uuid4().hex
        repository = await PostgresVideoProjectRepository.connect(database_url)
        runtime = VideoBuildRuntime(repository=repository)
        project, _version = await runtime.create_project(
            user_id=f"integration-user-{suffix}",
            title="Postgres integration",
            idempotency_key=f"create-{suffix}",
        )
        repeated, _version = await runtime.create_project(
            user_id=f"integration-user-{suffix}",
            title="Postgres integration",
            idempotency_key=f"create-{suffix}",
        )
        self.assertEqual(repeated.id, project.id)

        character = await runtime.add_artifact(MediaArtifactVersion(
            project_id=project.id,
            type="character",
            uri="postgres-test://character-v1",
        ))
        shot_candidate = MediaArtifactVersion(
            project_id=project.id,
            type="video",
            uri="postgres-test://shot-v1",
        )
        shot = await runtime.add_artifact(shot_candidate, [ArtifactDependency(
            project_id=project.id,
            source_version_id=character.id,
            target_version_id=shot_candidate.id,
            invalidation_policy=ArtifactInvalidationPolicy.HARD,
        )])
        current = await repository.get_project(project.id)
        preview = await runtime.preview_change(
            project_id=project.id,
            description="change the character wardrobe",
            target_artifact_version_ids=[character.id],
            idempotency_key=f"preview-{suffix}",
        )
        repeated_preview = await runtime.preview_change(
            project_id=project.id,
            description="change the character wardrobe",
            target_artifact_version_ids=[character.id],
            idempotency_key=f"preview-{suffix}",
        )
        self.assertEqual(repeated_preview.id, preview.id)
        self.assertEqual(set(preview.ids_for("rebuild")), {character.id, shot.id})
        build = await runtime.apply_rebuild(
            project_id=project.id,
            plan_id=preview.id,
            base_project_version_id=current.current_version_id,
            idempotency_key=f"build-{suffix}",
            session_id=f"session-{suffix}",
            user_id=f"integration-user-{suffix}",
        )
        self.assertEqual(build.status, "queued")
        await runtime.close()
        await repository.close()

        recovered_repository = await PostgresVideoProjectRepository.connect(database_url)
        recovered_runtime = VideoBuildRuntime(repository=recovered_repository)
        queued = await recovered_repository.get_build(project.id, build.id)
        self.assertEqual(queued.status, "queued")
        completed, version = await recovered_runtime.execute_build(
            project_id=project.id,
            build_id=build.id,
            executor=_FakeExecutor(),
        )
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.project_version_id, version.id)
        artifacts = await recovered_repository.current_artifacts(project.id)
        self.assertEqual(len(artifacts), 2)
        self.assertTrue(all(item.uri.startswith("postgres-test://") for item in artifacts))
        await recovered_runtime.close()
        await recovered_repository.close()

    async def test_initial_build_plan_and_steps_survive_repository_restart(self) -> None:
        database_url = os.environ["VIDEO_RUNTIME_DATABASE_URL"]
        suffix = uuid4().hex
        repository = await PostgresVideoProjectRepository.connect(database_url)
        runtime = VideoBuildRuntime(repository=repository)
        await runtime.plugins.load_directories([
            Path(__file__).resolve().parents[2] / "plugins",
        ])
        await load_workflow_skills(runtime.plugins, runtime.skills)
        project, version = await runtime.create_project(
            user_id=f"initial-user-{suffix}", title="Initial Postgres build",
        )
        plan = await runtime.plan_project(
            project_id=project.id,
            base_project_version_id=version.id,
            idempotency_key=f"plan-{suffix}",
            video_spec=VideoSpec.model_validate({
                "title": "Five seconds", "target_duration_seconds": 5,
                "characters": [],
                "shots": [{
                    "id": "one", "order": 1, "duration_seconds": 5,
                    "beat": "arrive", "visual_prompt": "A train arrives",
                }],
                "audio": {"bgm_prompt": "", "subtitles": False},
            }),
        )
        build = await runtime.start_build(
            project_id=project.id, plan_id=plan.id,
            base_project_version_id=version.id,
            idempotency_key=f"build-{suffix}",
        )
        self.assertTrue(await repository.list_build_steps(project.id, build.id))
        await runtime.close()
        await repository.close()

        recovered_repository = await PostgresVideoProjectRepository.connect(database_url)
        recovered = VideoBuildRuntime(repository=recovered_repository)

        async def validate(*, build_id, artifact):
            if artifact.type not in {"video_clip", "timeline", "video_assembled"}:
                return []
            return [ValidationResult(
                project_id=project.id, build_id=build_id,
                artifact_version_id=artifact.id, validator_id="postgres-test", passed=True,
            )]

        recovered.validate_artifact = validate
        completed, committed = await recovered.execute_build(
            project_id=project.id, build_id=build.id, executor=_FakeExecutor(),
        )
        self.assertEqual(completed.status, "completed")
        self.assertGreater(completed.actual_cost, 0)
        self.assertEqual(
            (await recovered_repository.get_build(project.id, build.id)).actual_cost,
            completed.actual_cost,
        )
        self.assertEqual((await recovered_repository.get_project(project.id)).current_version_id, committed.id)
        steps = await recovered_repository.list_build_steps(project.id, build.id)
        self.assertTrue(all(step.status == "completed" for step in steps))
        media_plan = await recovered.preview_edits(
            project_id=project.id,
            base_project_version_id=committed.id,
            edits=[{"type": "replace_music", "prompt": "warm acoustic guitar"}],
            idempotency_key=f"music-plan-{suffix}",
        )
        self.assertEqual(media_plan.estimated_cost, 0.11)
        self.assertEqual(
            {item.step_id for item in media_plan.items if item.action == "create"},
            {"bgm", "mix-bgm"},
        )
        media_build = await recovered.start_build(
            project_id=project.id,
            plan_id=media_plan.id,
            base_project_version_id=committed.id,
            idempotency_key=f"music-build-{suffix}",
        )
        completed_media, media_version = await recovered.execute_build(
            project_id=project.id,
            build_id=media_build.id,
            executor=_FakeExecutor(),
        )
        self.assertEqual(completed_media.actual_cost, 0.11)
        self.assertNotEqual(media_version.id, committed.id)
        selected = await recovered_repository.current_artifacts(project.id)
        self.assertTrue(any(item.type == "audio_bgm" for item in selected))
        self.assertTrue(any(item.type == "video_mixed" for item in selected))
        await recovered.close()
        await recovered_repository.close()

    async def test_cancellation_absorbed_by_a_step_still_stops_the_rebuild(self) -> None:
        database_url = os.environ["VIDEO_RUNTIME_DATABASE_URL"]
        suffix = uuid4().hex
        repository = await PostgresVideoProjectRepository.connect(database_url)
        runtime = VideoBuildRuntime(repository=repository)
        project, _version = await runtime.create_project(
            user_id=f"integration-user-{suffix}",
            title="Postgres cancellation",
            idempotency_key=f"create-{suffix}",
        )
        character = await runtime.add_artifact(MediaArtifactVersion(
            project_id=project.id,
            type="character",
            uri="postgres-test://character-v1",
        ))
        shot_candidate = MediaArtifactVersion(
            project_id=project.id,
            type="video",
            uri="postgres-test://shot-v1",
        )
        await runtime.add_artifact(shot_candidate, [ArtifactDependency(
            project_id=project.id,
            source_version_id=character.id,
            target_version_id=shot_candidate.id,
            invalidation_policy=ArtifactInvalidationPolicy.HARD,
        )])
        before = await repository.get_project(project.id)
        preview = await runtime.preview_change(
            project_id=project.id,
            description="change the character wardrobe",
            target_artifact_version_ids=[character.id],
            idempotency_key=f"preview-{suffix}",
        )
        build = await runtime.apply_rebuild(
            project_id=project.id,
            plan_id=preview.id,
            base_project_version_id=before.current_version_id,
            idempotency_key=f"build-{suffix}",
        )
        executor = _CancellationAbsorbingExecutor()
        runtime._default_executor = executor
        runtime._schedule_build(build)
        await executor.started.wait()

        cancelled = await runtime.cancel_build(project_id=project.id, build_id=build.id)

        self.assertEqual(cancelled.status, "cancelled")
        stopped = await repository.get_build(project.id, build.id)
        self.assertEqual(stopped.status, "cancelled")
        self.assertIsNone(stopped.project_version_id)
        self.assertEqual(executor.calls, 1)
        after = await repository.get_project(project.id)
        self.assertEqual(after.current_version_id, before.current_version_id)
        await runtime.close()
        await repository.close()
