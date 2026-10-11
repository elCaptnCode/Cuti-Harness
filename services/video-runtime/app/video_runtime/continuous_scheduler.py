"""Event-driven execution of Cuti PlanPatches; creative decisions stay in DeepSeek."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING

from .plan_utils import topological_steps
from .models import Build, BuildStep, MediaArtifactVersion, PlanCheckpoint, ProjectVersion, RebuildPlan, RebuildPlanItem, now, uid
from .repository import PlanRevisionConflict
from .moderation_recovery import failure_details
from .validation_feedback import task_failure_details, validation_rejections

if TYPE_CHECKING:
    from .runtime import BuildStepExecutor, VideoBuildRuntime


async def _notify_agent(
    runtime: VideoBuildRuntime, build: Build, plan: RebuildPlan,
    states: dict[str, BuildStep], completed: dict[str, MediaArtifactVersion], *, force: bool = False,
) -> PlanCheckpoint | None:
    """Persist one outstanding snapshot; later terminal events survive in Step state."""
    checkpoints = await runtime.repo.list_build_checkpoints(build.project_id, build.id)
    outstanding = next((item for item in checkpoints if item.status in {"pending", "planning", "failed"}), None)
    repairing = force and build.status == "failed"
    if outstanding is not None and not repairing:
        return outstanding
    latest = await runtime.repo.get_plan(plan.id)
    if latest.current_revision != plan.current_revision:
        return
    terminal = sorted(
        (item.plan_step_id, item.status, item.result_artifact_version_id or "")
        for item in states.values() if item.status in {"completed", "failed"}
    )
    if not terminal and not force:
        return
    phase = "task-update:" + hashlib.sha256(json.dumps(terminal).encode()).hexdigest()[:24]
    if force:
        phase = f"repair:{plan.current_revision}" if repairing else f"live:{plan.current_revision}:{uid()}"
    if any(item.phase == phase for item in checkpoints):
        return next(item for item in checkpoints if item.phase == phase) if repairing else None
    from .runtime import _checkpoint_artifact_summary, _checkpoint_task_parameters

    spec = await runtime.repo.get_video_spec_revision(plan.video_spec_revision_id)
    session_id, user_id = build.session_id, build.user_id
    if not session_id or not user_id:
        project = await runtime.repo.get_project(build.project_id)
        binding = await runtime.repo.latest_session_binding(build.project_id, project.user_id)
        session_id, user_id = binding.session_id, binding.user_id
    rejections = await validation_rejections(runtime, build.project_id, plan, states)
    snapshot = [
        {
            "step_id": item.step_id,
            "objective": item.objective,
            "capability": item.capability,
            "depends_on": item.depends_on,
            "parameters": _checkpoint_task_parameters(item.parameters),
            "status": states[item.step_id].status,
            "error": states[item.step_id].error,
            "failure": task_failure_details(
                plan, item, states[item.step_id], states, rejections,
            ),
            "artifact_version_id": states[item.step_id].result_artifact_version_id,
        }
        for item in plan.items
    ]
    return await runtime.repo.create_checkpoint(PlanCheckpoint(
        project_id=build.project_id, build_id=build.id, plan_id=plan.id,
        workflow_id=plan.workflow_id, session_id=session_id, user_id=user_id,
        phase=phase, next_phase="agent_execution", planning_mode="agentic",
        base_plan_revision=plan.current_revision, base_spec_revision=spec.revision,
        artifact_version_ids=[item.id for item in completed.values()],
        artifact_summaries=[_checkpoint_artifact_summary(item) for item in completed.values()],
        resolved_sections=spec.resolved_sections, unresolved_sections=spec.unresolved_sections,
        planner_instruction=(
            "A task completed or failed. Other tasks may still be running. Inspect current "
            "Build steps before submitting a PlanPatch. You may add tasks (including dependencies "
            "on existing or newly added tasks), cancel pending tasks and their pending dependants, "
            "or acknowledge this update with no added tasks while existing work continues. "
            "Never repeat completed work. Repair failed work with corrected add_tasks and "
            "replace_failed_task_ids mapping old task IDs to new client_keys; Runtime clones "
            "and rewires pending descendants. Do not use unchanged build retry for invalid parameters. Failed tasks "
            "are not automatically resubmitted. Completion requires no active tasks.\n"
            "When editing, the previous selected video is not proof that the NEW request is "
            "complete. Produce the requested edited/new artifact before goal_satisfied. "
            "Preserve unchanged selections; use the target artifact's logical output ID when "
            "replacing it. Existing media are available context, not automatic video references. "
            "Choose reference inputs explicitly from the user's intent and capability schema.\n"
            "Current user goal: " + (plan.project_intent.brief if plan.project_intent else "") + "\n"
            "Requested edits: " + json.dumps(
                plan.project_intent.constraints.get("edits", []) if plan.project_intent else [],
                ensure_ascii=False,
            ) + "\n"
            "Task snapshot at notification time:\n" + json.dumps(snapshot, ensure_ascii=False)
        ),
    ))


async def execute_continuous_build(
    runtime: VideoBuildRuntime, build: Build, executor: BuildStepExecutor,
) -> tuple[Build, ProjectVersion | None]:
    """Run ready tasks without a batch barrier and reload the plan after every wake."""
    wake = runtime._continuous_wakes.setdefault(build.id, asyncio.Event())
    active: dict[str, asyncio.Task[None]] = {}
    completed: dict[str, MediaArtifactVersion] = {}

    async def run_step(planned: RebuildPlanItem, state: BuildStep, plan: RebuildPlan) -> None:
        try:
            if planned.action == "reuse":
                completed[planned.step_id] = await runtime.repo.get_artifact(
                    build.project_id, planned.artifact_version_id,
                )
                state.result_artifact_version_id = completed[planned.step_id].id
            elif planned.action == "validate":
                # A requeued step still points at its previous attempt's report; a
                # failed attempt must point only at the report it staged itself.
                state.result_artifact_version_id = None
                results = []
                for dependency in planned.depends_on:
                    if dependency in completed:
                        results.extend(await runtime.validate_artifact(
                            build_id=build.id, artifact=completed[dependency],
                        ))
                report = await runtime.repo.stage_artifact(MediaArtifactVersion(
                    project_id=build.project_id,
                    artifact_id=planned.output_artifact_id or f"{build.project_id}:{planned.step_id}",
                    type="validation_result", status="draft",
                    metadata={"validation_results": [result.model_dump(mode="json") for result in results]},
                ))
                state.result_artifact_version_id = report.id
                if not results or any(not result.passed for result in results):
                    raise ValueError("Artifact validation failed: " + "; ".join(
                        issue for result in results for issue in result.issues
                    ))
                completed[planned.step_id] = report
            else:
                await runtime._execute_initial_create_step(
                    build=build, planned=planned, state=state, completed=completed,
                    executor=executor,
                    # Retry transport inside the provider adapter; task failure
                    # must reach the Agent before another generation submission.
                    retry_limit=0,
                )
                return
            state.status = "completed"
            state.completed_at = now()
            await runtime.repo.update_build_step(state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.status = "failed"
            state.error = str(exc) or repr(exc)
            state.completed_at = now()
            await runtime.repo.update_build_step(state)

    try:
        while True:
            wake.clear()
            current = await runtime.repo.get_build(build.project_id, build.id)
            if current.status in {"cancelled", "failed", "completed"}:
                return current, None
            plan = await runtime.repo.get_plan(build.plan_id)
            states = {item.plan_step_id: item for item in await runtime.repo.list_build_steps(
                build.project_id, build.id,
            )}
            for step_id, task in list(active.items()):
                if task.done() and states[step_id].status in {"completed", "failed", "cancelled"}:
                    task.result()
                    del active[step_id]
            for step_id, state in states.items():
                if state.status == "completed" and state.result_artifact_version_id:
                    if step_id not in completed:
                        completed[step_id] = await runtime.repo.get_artifact(
                            build.project_id, state.result_artifact_version_id,
                        )
            blocked = {
                item.step_id for item in plan.items
                if (details := failure_details(plan, item, states[item.step_id], states))
                and details["requires_user_action"] and not item.superseded_by
            }
            for item in topological_steps(plan.items):
                if set(item.depends_on) & blocked and states[item.step_id].status == "pending":
                    blocked.add(item.step_id)
                    state = states[item.step_id]
                    state.status = "cancelled"
                    state.error = "Content moderation: user revision required"
                    state.completed_at = now()
                    await runtime.repo.update_build_step(state)
            if plan.next_checkpoint is None:
                if active or any(item.status in {"pending", "running", "waiting_external"}
                                 for item in states.values()):
                    raise ValueError("cannot commit a continuous build with active work")
                # All executable work has settled. Reuse the existing atomic
                # publication path, omitting failed history rather than rerunning it.
                finished = plan.model_copy(update={"items": [
                    item for item in plan.items if states[item.step_id].status == "completed"
                ]})
                return await runtime._execute_initial_build(build=current, plan=finished, executor=executor)

            done = [item for item in plan.items if states[item.step_id].status == "completed"]
            await runtime.repo.update_build_progress(
                build.project_id, build.id, len(done) / max(1, len(plan.items) + 1),
                round(sum(item.estimated_cost for item in done), 6),
            )
            if blocked and not active:
                current = await runtime.repo.get_build(build.project_id, build.id)
                current.status = "waiting_agent"
                current.message = "Content moderation: user revision required"
                await runtime.repo.update_build(current)
            try:
                await _notify_agent(runtime, current, plan, states, completed)
            except PlanRevisionConflict:
                continue  # A concurrently committed patch owns the next snapshot.
            ordered = topological_steps(plan.items)
            for item in ordered:
                if len(active) >= runtime.max_parallel_generation_tasks:
                    break
                state = states[item.step_id]
                if item.step_id in active or state.status not in {"pending", "running", "waiting_external"}:
                    continue
                if any(states[dependency].status != "completed" for dependency in item.depends_on):
                    continue
                # Compare-and-set protects admission against a concurrent cancel.
                if state.status == "pending":
                    state = await runtime.repo.claim_build_step(build.project_id, build.id, item.step_id)
                    if state is None:
                        continue
                active[item.step_id] = asyncio.create_task(run_step(item, state, plan))

            if not active:
                # No polling worker is needed while waiting for Agent input.
                # resolve_checkpoint schedules a worker or sets this worker's wake.
                if wake.is_set():
                    continue
                return await runtime.repo.get_build(build.project_id, build.id), None
            waiter = asyncio.create_task(wake.wait())
            try:
                await asyncio.wait([*active.values(), waiter], return_when=asyncio.FIRST_COMPLETED)
            finally:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
    finally:
        for task in active.values():
            task.cancel()
        await asyncio.gather(*active.values(), return_exceptions=True)
