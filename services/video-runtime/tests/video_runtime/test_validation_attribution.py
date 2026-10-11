"""Validation-failure attribution in continuous task snapshots.

A failed validate task names each ArtifactVersion a validator rejected, the task
that produced it, the validator and its findings, read back from the report the
scheduler persists. FakePlanExecutor stands in for providers and an injected
``validate_artifact`` stands in for validators; no test needs an API key.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI

from app.video_runtime.api import get_runtime
from app.video_runtime.builtin_plugins.continuity_validator import ContinuityValidatorPlugin
from app.video_runtime.checkpoint_coordinator import CheckpointCoordinator, refresh_checkpoint
from app.video_runtime.local_repository import LocalJsonVideoProjectRepository
from app.video_runtime.models import (
    BuildStep,
    CheckpointResolution,
    MediaArtifactVersion,
    ProjectIntent,
    RebuildPlanItem,
    ValidationResult,
)
from app.video_runtime.moderation_recovery import failure_details
from app.video_runtime.plugins import PluginContext
from test_initial_build import FakePlanExecutor, video_runtime


NOTIFIED = "Task snapshot at notification time:\n"
REFRESHED = "Current task snapshot (supersedes earlier notifications):\n"
SCENE_VALIDATOR = "cuti.continuity.scene-reference-isolation"
CHARACTER_VALIDATOR = "cuti.continuity.character-reference-identity"
PERSON = "rendered scene setting reference contains people or characters"
PRODUCT = "rendered scene setting reference contains a staged product"
WARDROBE = "rendered character setting reference does not match the locked character definition"
INCOMPLETE = "scene visual isolation verification is incomplete: timed out"


def task_in(instruction: str, marker: str, step_id: str) -> dict:
    """Return one task from the JSON snapshot that follows ``marker``."""
    snapshot = json.loads(instruction.split(marker, 1)[1])
    return next(item for item in snapshot if item["step_id"] == step_id)


def delivered_failure(prompt: str, step_id: str) -> dict:
    """Return one task failure from a delivered checkpoint prompt."""
    payload = json.loads(prompt.split("Checkpoint payload:\n", 1)[1].split("\nACTION REQUIRED", 1)[0])
    return task_in(payload["planner_instruction"], REFRESHED, step_id)["failure"]


def with_names(value: dict, names: dict[str, str]) -> dict:
    """Replace generated ids with stable names for snapshot comparison."""
    text = json.dumps(value, ensure_ascii=False)
    for raw, name in names.items():
        text = text.replace(raw, name)
    return json.loads(text)


class ValidationAttributionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.runtime = await video_runtime(max_parallel_generation_tasks=2)
        self.runtime.staged_planning_enabled = True
        self.runtime.continuous_plan_patch_enabled = True
        self.project, self.version = await self.runtime.create_project(user_id="u", title="Attribution")
        await self.runtime.bind_session(project_id=self.project.id, session_id="s", user_id="u")
        self.plan = await self.runtime.plan_project(
            project_id=self.project.id, base_project_version_id=self.version.id,
            project_intent=ProjectIntent(title="Attribution", brief="Make a video", workflow_id="seedance2"),
            idempotency_key="plan",
        )
        self.build = await self.runtime.start_build(
            project_id=self.project.id, plan_id=self.plan.id, base_project_version_id=self.version.id,
            idempotency_key="build", session_id="s", user_id="u",
        )
        self.executor = FakePlanExecutor()
        # Producing step id -> (validator_id, issues, metadata) of each result its artifact receives.
        self.verdicts: dict[str, list[tuple[str, list[str], dict]]] = {}
        # Producing step ids whose results name no ArtifactVersion.
        self.unnamed: set[str] = set()
        self.crash: Exception | None = None
        self.runtime.validate_artifact = self.validate
        await self.run_build()
        await self.resolve(await self.pending_checkpoint(), proposed_steps=[
            self.shot("clip-1"), self.shot("clip-2"),
        ])
        await self.run_build()

    async def asyncTearDown(self) -> None:
        await self.runtime.close()

    async def validate(self, *, build_id, artifact):
        if self.crash is not None:
            raise self.crash
        step_id = str(artifact.metadata.get("plan_step_id"))
        verdicts = self.verdicts.get(step_id, [("cuti.test.structure", [], {})])
        return [ValidationResult(
            project_id=self.project.id, build_id=build_id,
            artifact_version_id=None if step_id in self.unnamed else artifact.id,
            validator_id=validator_id, passed=not issues, issues=issues, metadata=metadata,
        ) for validator_id, issues, metadata in verdicts]

    async def run_build(self):
        return await self.runtime.execute_build(
            project_id=self.project.id, build_id=self.build.id, executor=self.executor,
        )

    async def pending_checkpoint(self):
        checkpoints = await self.runtime.repo.list_build_checkpoints(self.project.id, self.build.id)
        return next(item for item in reversed(checkpoints) if item.status in {"pending", "planning"})

    async def resolve(self, checkpoint, **kwargs):
        return await self.runtime.resolve_checkpoint(
            project_id=self.project.id, build_id=self.build.id, checkpoint_id=checkpoint.id,
            session_id="s", user_id="u",
            resolution=CheckpointResolution(
                base_plan_revision=checkpoint.base_plan_revision,
                base_spec_revision=checkpoint.base_spec_revision,
                idempotency_key=f"patch-{checkpoint.id}", video_spec_patch={}, **kwargs,
            ),
        )

    @staticmethod
    def shot(name: str) -> RebuildPlanItem:
        return RebuildPlanItem(
            step_id=name, action="create", capability="atomic.video.generate",
            parameters={"prompt": name, "model": "seedance-2.0", "generate_audio": True},
            depends_on=["intent"],
        )

    async def validate_task(self, name: str, dependencies: list[str]) -> None:
        """Commit a validate task as an accepted PlanPatch persists it, then run it.

        PlanPatch admission resolves capabilities through the Skill catalog, which
        does not list validators, so the plan item and its BuildStep are written
        directly.
        """
        item = RebuildPlanItem(
            step_id=name, action="validate", capability="cuti.continuity.validate",
            output_artifact_type="validation_result", depends_on=dependencies,
        )
        async with self.runtime.repo.lock:
            self.runtime.repo.plans[self.plan.id].items.append(item)
            self.runtime.repo.build_steps[self.build.id][name] = BuildStep(
                build_id=self.build.id, project_id=self.project.id, plan_step_id=name,
                action=item.action, capability=item.capability,
            )
        await self.run_build()

    async def states(self) -> dict[str, BuildStep]:
        return {item.plan_step_id: item for item in await self.runtime.repo.list_build_steps(
            self.project.id, self.build.id,
        )}

    async def refreshed_failure(self, step_id: str) -> dict | None:
        checkpoints = await self.runtime.repo.list_build_checkpoints(self.project.id, self.build.id)
        refreshed = await refresh_checkpoint(self.runtime, checkpoints[-1])
        return task_in(refreshed.planner_instruction, REFRESHED, step_id)["failure"]

    async def unattributed_failure(self, step_id: str) -> dict | None:
        """The failure payload the Runtime reported before attribution existed."""
        plan = await self.runtime.repo.get_plan(self.plan.id)
        states = await self.states()
        item = next(item for item in plan.items if item.step_id == step_id)
        return failure_details(plan, item, states[step_id], states)

    async def test_failed_validate_task_names_rejected_artifact_producer_and_validator(self):
        self.verdicts["clip-1"] = [(SCENE_VALIDATOR, [PERSON], {})]
        await self.validate_task("validate-1", ["clip-1", "clip-2"])
        states = await self.states()
        self.assertEqual(states["validate-1"].status, "failed")
        # Acknowledge the outstanding notification so the scheduler persists a new one.
        await self.resolve(await self.pending_checkpoint())
        await self.run_build()
        checkpoint = await self.pending_checkpoint()

        notified = task_in(checkpoint.planner_instruction, NOTIFIED, "validate-1")["failure"]
        refreshed = task_in(
            (await refresh_checkpoint(self.runtime, checkpoint)).planner_instruction,
            REFRESHED, "validate-1",
        )["failure"]
        self.assertEqual(notified, refreshed)
        expected = json.loads(
            (Path(__file__).parent / "snapshots" / "quality-rejected.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(with_names(refreshed, {
            states["clip-1"].result_artifact_version_id: "<clip-1 version>",
        }), expected)
        self.assertNotIn(states["clip-2"].result_artifact_version_id, json.dumps(refreshed))

    async def test_redelivered_checkpoint_repeats_the_same_attribution(self):
        self.verdicts["clip-1"] = [(SCENE_VALIDATOR, [PERSON], {})]
        await self.validate_task("validate-1", ["clip-1"])

        class Session:
            prompts: list[str] = []

            async def prompt(self, session_id, prompt, *, mode):
                self.prompts.append(prompt)

        session = Session()
        coordinator = CheckpointCoordinator(self.runtime, session)
        self.assertTrue(await coordinator.run_once())
        checkpoint = await self.pending_checkpoint()
        await self.runtime.repo.fail_checkpoint_delivery(
            checkpoint.id, "turn ended without a PlanPatch", expected_attempt=1,
        )
        self.assertTrue(await coordinator.run_once())

        first, second = (delivered_failure(prompt, "validate-1") for prompt in session.prompts)
        self.assertEqual(first, second)
        self.assertEqual(first["category"], "quality_rejected")
        inspected = await self.runtime.inspect_checkpoint(
            project_id=self.project.id, build_id=self.build.id, checkpoint_id=checkpoint.id,
        )
        self.assertEqual(task_in(inspected.planner_instruction, REFRESHED, "validate-1")["failure"], first)

    async def test_attribution_survives_repository_restart(self):
        self.verdicts["clip-1"] = [(SCENE_VALIDATOR, [PERSON], {})]
        await self.validate_task("validate-1", ["clip-1"])
        checkpoint = await self.pending_checkpoint()
        before = await self.refreshed_failure("validate-1")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            stored = LocalJsonVideoProjectRepository(path)
            for name, value in vars(self.runtime.repo).items():
                if name != "lock":
                    setattr(stored, name, value)
            await stored.close()
            restarted = await video_runtime(max_parallel_generation_tasks=2)
            restarted.repo = LocalJsonVideoProjectRepository(path)
            after = task_in(
                (await refresh_checkpoint(restarted, checkpoint)).planner_instruction,
                REFRESHED, "validate-1",
            )["failure"]
            await restarted.close()

        self.assertEqual(after, before)
        self.assertEqual(after["category"], "quality_rejected")

    async def test_each_validate_task_keeps_the_findings_of_its_own_report(self):
        self.verdicts["clip-1"] = [(SCENE_VALIDATOR, [PERSON], {})]
        await self.validate_task("validate-1", ["clip-1"])
        self.verdicts["clip-1"] = [(CHARACTER_VALIDATOR, [WARDROBE], {})]
        await self.validate_task("validate-2", ["clip-1"])
        await self.validate_task("validate-3", ["clip-2"])
        states = await self.states()
        clip_1 = states["clip-1"].result_artifact_version_id

        self.assertEqual((await self.refreshed_failure("validate-1"))["rejected_artifacts"], [{
            "artifact_version_id": clip_1, "producer_task_id": "clip-1",
            "validator_id": SCENE_VALIDATOR, "issues": [PERSON],
        }])
        self.assertEqual((await self.refreshed_failure("validate-2"))["rejected_artifacts"], [{
            "artifact_version_id": clip_1, "producer_task_id": "clip-1",
            "validator_id": CHARACTER_VALIDATOR, "issues": [WARDROBE],
        }])
        self.assertEqual(states["validate-3"].status, "completed")
        self.assertIsNone(await self.refreshed_failure("validate-3"))

    async def test_rerun_validate_task_reports_only_its_latest_report(self):
        self.verdicts["clip-1"] = [(SCENE_VALIDATOR, [PERSON], {})]
        await self.validate_task("validate-1", ["clip-1"])
        first_report = (await self.states())["validate-1"].result_artifact_version_id
        build = await self.runtime.repo.get_build(self.project.id, self.build.id)
        build.status = "failed"
        await self.runtime.repo.update_build(build)
        await self.runtime.retry_failed_build(project_id=self.project.id, build_id=self.build.id)
        self.verdicts["clip-1"] = [(SCENE_VALIDATOR, [PRODUCT], {})]
        await self.run_build()

        states = await self.states()
        self.assertEqual(states["validate-1"].status, "failed")
        self.assertNotEqual(states["validate-1"].result_artifact_version_id, first_report)
        failure = await self.refreshed_failure("validate-1")
        self.assertEqual([entry["issues"] for entry in failure["rejected_artifacts"]], [[PRODUCT]])
        self.assertEqual(self.executor.calls.count("clip-1"), 1)

    async def test_rerun_that_fails_before_staging_does_not_report_the_previous_findings(self):
        self.verdicts["clip-1"] = [(SCENE_VALIDATOR, [PERSON], {})]
        await self.validate_task("validate-1", ["clip-1"])
        build = await self.runtime.repo.get_build(self.project.id, self.build.id)
        build.status = "failed"
        await self.runtime.repo.update_build(build)
        await self.runtime.retry_failed_build(project_id=self.project.id, build_id=self.build.id)
        self.crash = RuntimeError("validator plugin crashed")
        await self.run_build()

        state = (await self.states())["validate-1"]
        self.assertEqual((state.status, state.error), ("failed", "validator plugin crashed"))
        self.assertIsNone(state.result_artifact_version_id)
        failure = await self.refreshed_failure("validate-1")
        self.assertEqual(failure, await self.unattributed_failure("validate-1"))
        self.assertNotIn("rejected_artifacts", failure)

    async def test_producer_is_the_dependency_that_holds_the_rejected_version(self):
        clip_1 = (await self.states())["clip-1"].result_artifact_version_id
        # An earlier plan item that also holds clip-1's version, as a reuse step does.
        async with self.runtime.repo.lock:
            self.runtime.repo.plans[self.plan.id].items.insert(0, RebuildPlanItem(
                step_id="reuse-clip-1", action="reuse", artifact_version_id=clip_1,
            ))
            self.runtime.repo.build_steps[self.build.id]["reuse-clip-1"] = BuildStep(
                build_id=self.build.id, project_id=self.project.id, plan_step_id="reuse-clip-1",
                action="reuse", status="completed", result_artifact_version_id=clip_1,
            )
        self.verdicts["clip-1"] = [(SCENE_VALIDATOR, [PERSON], {})]
        await self.validate_task("validate-1", ["clip-1"])

        failure = await self.refreshed_failure("validate-1")
        self.assertEqual(
            [entry["producer_task_id"] for entry in failure["rejected_artifacts"]], ["clip-1"],
        )

    async def test_every_conclusive_rejection_in_one_report_is_listed(self):
        self.verdicts["clip-1"] = [
            (SCENE_VALIDATOR, [PERSON], {}), (CHARACTER_VALIDATOR, [WARDROBE], {}),
        ]
        self.verdicts["clip-2"] = [(SCENE_VALIDATOR, [PRODUCT], {})]
        await self.validate_task("validate-1", ["clip-1", "clip-2"])
        states = await self.states()

        failure = await self.refreshed_failure("validate-1")
        self.assertEqual(
            [(entry["producer_task_id"], entry["validator_id"], entry["issues"])
             for entry in failure["rejected_artifacts"]],
            [("clip-1", SCENE_VALIDATOR, [PERSON]), ("clip-1", CHARACTER_VALIDATOR, [WARDROBE]),
             ("clip-2", SCENE_VALIDATOR, [PRODUCT])],
        )
        self.assertEqual(
            {entry["artifact_version_id"] for entry in failure["rejected_artifacts"]},
            {states["clip-1"].result_artifact_version_id, states["clip-2"].result_artifact_version_id},
        )

    async def test_rejection_naming_no_artifact_keeps_the_existing_payload(self):
        self.unnamed.add("clip-1")
        self.verdicts["clip-1"] = [(SCENE_VALIDATOR, [PERSON], {})]
        await self.validate_task("validate-1", ["clip-1"])

        failure = await self.refreshed_failure("validate-1")
        self.assertEqual(failure, await self.unattributed_failure("validate-1"))
        self.assertNotIn("rejected_artifacts", failure)

    async def test_unhandled_and_inconclusive_results_keep_the_existing_payload(self):
        self.verdicts["clip-1"] = []
        self.verdicts["clip-2"] = [(SCENE_VALIDATOR, [INCOMPLETE], {"inconclusive": True})]
        await self.validate_task("validate-1", ["clip-1"])
        await self.validate_task("validate-2", ["clip-2"])

        for step_id in ("validate-1", "validate-2"):
            with self.subTest(step_id=step_id):
                failure = await self.refreshed_failure(step_id)
                self.assertEqual(failure, await self.unattributed_failure(step_id))
                self.assertNotIn("rejected_artifacts", failure)
                self.assertNotEqual(failure["category"], "quality_rejected")

    async def test_inconclusive_result_beside_a_rejection_is_not_attributed(self):
        self.verdicts["clip-1"] = [(SCENE_VALIDATOR, [PERSON], {})]
        self.verdicts["clip-2"] = [(SCENE_VALIDATOR, [INCOMPLETE], {"inconclusive": True})]
        await self.validate_task("validate-1", ["clip-1", "clip-2"])
        clip_1 = (await self.states())["clip-1"].result_artifact_version_id

        failure = await self.refreshed_failure("validate-1")
        self.assertEqual(failure["category"], "quality_rejected")
        self.assertEqual(
            [entry["artifact_version_id"] for entry in failure["rejected_artifacts"]], [clip_1],
        )

    async def test_validator_text_classified_as_moderation_keeps_the_moderation_payload(self):
        self.verdicts["clip-1"] = [(SCENE_VALIDATOR, ["the pose violates the storyboard"], {})]
        await self.validate_task("validate-1", ["clip-1"])

        failure = await self.refreshed_failure("validate-1")
        self.assertEqual(failure, await self.unattributed_failure("validate-1"))
        self.assertEqual(failure["category"], "content_moderation")

    async def test_studio_task_list_reports_the_same_attribution(self):
        from app.video_runtime.deepseek_bff import get_deepseek_client, router as bff_router

        self.verdicts["clip-1"] = [(SCENE_VALIDATOR, [PERSON], {})]
        await self.validate_task("validate-1", ["clip-1"])

        class History:
            async def history(self, *args, **kwargs):
                return {"events": [], "hasMore": False}

        app = FastAPI()
        app.include_router(bff_router)
        app.dependency_overrides[get_runtime] = lambda: self.runtime
        app.dependency_overrides[get_deepseek_client] = lambda: History()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
            headers={"X-Video-User-Id": "u"},
        ) as client:
            response = await client.get(f"/v2/runs/{self.project.id}")
        self.assertEqual(response.status_code, 200, response.text)
        task = next(item for item in response.json()["data"]["tasks"] if item["objective"] == "validate 1")
        self.assertEqual(task["failure"]["category"], "quality_rejected")
        self.assertEqual(
            [entry["producer_task_id"] for entry in task["failure"]["rejected_artifacts"]], ["clip-1"],
        )
        self.assertEqual(task["failure"], await self.refreshed_failure("validate-1"))


class ContinuityValidatorInconclusiveTest(unittest.IsolatedAsyncioTestCase):
    """Exception-derived results are marked inconclusive; verdicts are not."""

    VALIDATOR = "app.video_runtime.builtin_plugins.continuity_validator"

    @staticmethod
    def reference(role: str, prompt: str) -> MediaArtifactVersion:
        return MediaArtifactVersion(
            project_id="project-1", artifact_id=role, type="image",
            uri=f"https://cdn.example.test/{role}.webp",
            metadata={
                "generation_parameters": {"prompt": prompt, "artifact_role": role},
                "resolved_generation_parameters": {"prompt": prompt},
                "final_prompt": prompt,
                "skill_prompt_applied": False,
            },
        )

    async def validate(self, artifact: MediaArtifactVersion) -> ValidationResult:
        results = await ContinuityValidatorPlugin().validate_artifact(
            PluginContext(project_id="project-1", build_id="build-1"), artifact,
        )
        self.assertEqual(len(results), 1)
        return results[0]

    async def test_failed_inspections_are_inconclusive(self):
        scene = self.reference(
            "scene_setting_reference",
            "Create exactly one unoccupied environment-only setting reference. "
            "Render a single coherent wide architectural view.",
        )
        character = self.reference(
            "character_setting_reference",
            "Create ONLY one clean recurring-character identity reference sheet. "
            "Do not include the advertised product.",
        )
        final_video = MediaArtifactVersion(
            project_id="project-1", artifact_id="final", type="final_video",
            uri="https://cdn.example.test/final.mp4",
            metadata={"generation_parameters": {"final_output": True}},
        )
        cases = [
            ("scene", scene, "_inspect_scene_reference"),
            ("character", character, "_inspect_character_reference"),
            ("ffprobe", final_video, "_video_info"),
        ]
        for name, artifact, inspector in cases:
            with self.subTest(name), patch.dict(
                "os.environ", {"VIDEO_SCENE_REFERENCE_VISUAL_VALIDATION_ENABLED": "true"},
            ), patch(
                f"{self.VALIDATOR}.{inspector}",
                new=AsyncMock(side_effect=httpx.ReadTimeout("timed out")),
            ):
                result = await self.validate(artifact)
                self.assertFalse(result.passed)
                self.assertEqual(len(result.issues), 1)
                self.assertIs(result.metadata.get("inconclusive"), True)

    async def test_inspection_failing_after_a_verdict_issue_is_not_inconclusive(self):
        scene = self.reference(
            "scene_setting_reference",
            "Create exactly one unoccupied environment-only setting reference. "
            "Render a single coherent wide architectural view.",
        )
        # The verdict lacks product_present, so reading it raises after one issue is recorded.
        with patch.dict(
            "os.environ", {"VIDEO_SCENE_REFERENCE_VISUAL_VALIDATION_ENABLED": "true"},
        ), patch(
            f"{self.VALIDATOR}._inspect_scene_reference",
            new=AsyncMock(return_value={"people_present": True}),
        ):
            result = await self.validate(scene)
        self.assertEqual(len(result.issues), 2)
        self.assertNotIn("inconclusive", result.metadata)

    async def test_visual_verdict_is_not_inconclusive(self):
        scene = self.reference(
            "scene_setting_reference",
            "Create exactly one unoccupied environment-only setting reference. "
            "Render a single coherent wide architectural view.",
        )
        verdict = {
            "people_present": True, "product_present": False,
            "contact_sheet_present": False, "reason": "A person stands on the platform.",
        }
        with patch.dict(
            "os.environ", {"VIDEO_SCENE_REFERENCE_VISUAL_VALIDATION_ENABLED": "true"},
        ), patch(f"{self.VALIDATOR}._inspect_scene_reference", new=AsyncMock(return_value=verdict)):
            result = await self.validate(scene)
        self.assertFalse(result.passed)
        self.assertNotIn("inconclusive", result.metadata)


if __name__ == "__main__":
    unittest.main()
