"""Attribute failed validate tasks to the ArtifactVersions their validators rejected."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.utils.error_classification import FailureCategory

from .models import BuildStep, RebuildPlan, RebuildPlanItem, ValidationResult
from .moderation_recovery import failure_details, is_moderation

if TYPE_CHECKING:
    from .runtime import VideoBuildRuntime


async def validation_rejections(
    runtime: VideoBuildRuntime, project_id: str, plan: RebuildPlan,
    states: dict[str, BuildStep],
) -> dict[str, list[dict[str, Any]]]:
    """Read conclusive rejections back from each failed validate task's report.

    The continuous scheduler stages every ValidationResult of an attempt in a
    ``validation_result`` report and points the BuildStep at it before raising, so
    attribution is derived from that durable report, never from the joined error
    text. Omitted results: passing ones, ones naming no ArtifactVersion, and ones a
    validator marks ``metadata["inconclusive"]`` because it reached no verdict.

    Args:
        runtime: Runtime whose repository holds the reports.
        project_id: Project that owns the Build.
        plan: Plan whose items identify validate tasks and their dependencies.
        states: BuildSteps keyed by plan step id.

    Returns:
        Rejections keyed by validate step id; steps without one are absent. Each
        entry's ``producer_task_id`` is the validate task's dependency whose BuildStep
        holds the rejected version (the first in ``depends_on`` order), or None when
        no dependency holds it.
    """
    from .runtime import _bounded_checkpoint_value

    rejections: dict[str, list[dict[str, Any]]] = {}
    for item in plan.items:
        state = states.get(item.step_id)
        if (
            item.action != "validate" or state is None
            or state.status != "failed" or not state.result_artifact_version_id
        ):
            continue
        producers: dict[str, str] = {}
        for dependency in item.depends_on:
            produced = states.get(dependency)
            if produced is not None and produced.result_artifact_version_id:
                producers.setdefault(produced.result_artifact_version_id, dependency)
        report = await runtime.repo.get_artifact(project_id, state.result_artifact_version_id)
        found = [
            {
                "artifact_version_id": result.artifact_version_id,
                "producer_task_id": producers.get(result.artifact_version_id),
                "validator_id": result.validator_id,
                "issues": _bounded_checkpoint_value(list(result.issues)),
            }
            for result in map(
                ValidationResult.model_validate,
                report.metadata.get("validation_results") or [],
            )
            if not result.passed and result.artifact_version_id
            and not result.metadata.get("inconclusive")
        ]
        if found:
            rejections[item.step_id] = found
    return rejections


def task_failure_details(
    plan: RebuildPlan, item: RebuildPlanItem, state: BuildStep,
    states: dict[str, BuildStep], rejections: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Return the agent-visible failure of one task, attributing validator rejections.

    A validate task with rejections reports category ``quality_rejected``, trigger
    source ``validator`` and ``rejected_artifacts``. A task whose error classifies as
    content moderation keeps the moderation payload unchanged, so its category agrees
    with the classification that moderation repair enforcement applies to the same
    error text.

    Args:
        plan: Plan that owns the task.
        item: Plan item of the task.
        state: BuildStep of the task.
        states: BuildSteps keyed by plan step id.
        rejections: Result of :func:`validation_rejections` for the same plan and states.

    Returns:
        The failure payload, or None when the task recorded no error.
    """
    details = failure_details(plan, item, state, states)
    found = rejections.get(item.step_id)
    if details is None or not found or is_moderation(state.error):
        return details
    return {
        **details,
        "category": FailureCategory.QUALITY_REJECTED.value,
        "trigger_source": "validator",
        "rejected_artifacts": found,
    }
