from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.orchestration.skills.models import (
    ResolvedSkillRef,
    SkillConstraintContract,
    SkillContext,
)
from app.domain.skills.lock import SkillLock


def uid() -> str:
    return str(uuid4())


def now() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactInvalidationPolicy(StrEnum):
    HARD = "hard"
    VALIDATE = "validate"
    SOFT = "soft"
    NONE = "none"


class Project(BaseModel):
    id: str = Field(default_factory=uid)
    user_id: str
    title: str
    status: str = "active"
    current_version_id: str = ""
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class ProjectSessionBinding(BaseModel):
    project_id: str
    session_id: str
    user_id: str
    created_at: datetime = Field(default_factory=now)


class ProjectSkillLock(SkillLock):
    """Versioned Skill selection owned by a long-lived video project."""

    updated_at: datetime = Field(default_factory=now)


class ProjectVersion(BaseModel):
    id: str = Field(default_factory=uid)
    project_id: str
    parent_version_id: str | None = None
    timeline_version_id: str | None = None
    selections: dict[str, str] = Field(default_factory=dict)
    change_request_id: str | None = None
    video_spec_revision_id: str | None = None
    created_at: datetime = Field(default_factory=now)


class MediaArtifactVersion(BaseModel):
    id: str = Field(default_factory=uid)
    artifact_id: str = Field(default_factory=uid)
    project_id: str
    type: str
    version: int = 1
    status: Literal["draft", "ready", "approved", "rejected", "stale", "superseded"] = "ready"
    uri: str | None = None
    title: str = ""
    summary: str = ""
    content_digest: str = ""
    generation_spec_digest: str = ""
    provider_id: str | None = None
    provider_version: str | None = None
    plugin_id: str | None = None
    plugin_version: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now)


class ArtifactDependency(BaseModel):
    id: str = Field(default_factory=uid)
    project_id: str
    source_version_id: str
    target_version_id: str
    relation: str = "derived_from"
    invalidation_policy: ArtifactInvalidationPolicy = ArtifactInvalidationPolicy.HARD
    created_at: datetime = Field(default_factory=now)


class ChangeRequest(BaseModel):
    id: str = Field(default_factory=uid)
    project_id: str
    base_project_version_id: str
    description: str
    target_artifact_version_ids: list[str] = Field(default_factory=list)
    edits: list[dict[str, Any]] = Field(default_factory=list)
    proposed_video_spec: "VideoSpec | None" = None
    created_at: datetime = Field(default_factory=now)


class MediaCapabilityInputContract(BaseModel):
    """One project-artifact input accepted by a workflow-free media capability."""

    model_config = ConfigDict(extra="forbid")

    role: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    artifact_types: list[str] = Field(min_length=1)
    parameter: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    required: bool = True
    multiple: bool = False
    description: str = ""


class MediaCapabilityContract(BaseModel):
    """Plugin-owned contract used to build post-production plans dynamically."""

    model_config = ConfigDict(extra="forbid")

    capability: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,127}$")
    description: str = Field(min_length=1)
    inputs: list[MediaCapabilityInputContract] = Field(default_factory=list)
    output_artifact_type: str = Field(min_length=1, max_length=100)
    replaces_input_role: str | None = None
    estimated_cost: float = Field(default=0.0, ge=0)
    skill_id: str | None = None
    parameters_schema: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_roles(self):
        roles = [item.role for item in self.inputs]
        if len(roles) != len(set(roles)):
            raise ValueError("media capability input roles must be unique")
        if self.replaces_input_role and self.replaces_input_role not in roles:
            raise ValueError("replaces_input_role must name a declared input role")
        return self


class MediaEditInput(BaseModel):
    """Reference either a selected project ArtifactVersion or an earlier edit step."""

    model_config = ConfigDict(extra="forbid")

    role: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    artifact_version_id: str | None = None
    operation_step_id: str | None = None

    @model_validator(mode="after")
    def require_one_source(self):
        if bool(self.artifact_version_id) == bool(self.operation_step_id):
            raise ValueError(
                "media edit input requires exactly one of artifact_version_id or operation_step_id"
            )
        return self


class MediaEditOperation(BaseModel):
    """One Agent-proposed step constrained by an installed Media Plugin contract."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    capability: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,127}$")
    inputs: list[MediaEditInput] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    title: str = Field(default="", max_length=200)


class VideoCharacterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    appearance: str = Field(min_length=1)
    clothing: str = ""
    personality: str = ""
    voice: str = ""


class VideoShotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=100)
    order: int = Field(ge=1)
    duration_seconds: float = Field(gt=0, le=30)
    beat: str = Field(min_length=1)
    visual_prompt: str = Field(min_length=1)
    narration: str = ""
    character_ids: list[str] = Field(default_factory=list)
    reference_asset_ids: list[str] = Field(default_factory=list)
    transition: str = "cut"


class VideoAudioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    narration_voice: str = "default"
    bgm_prompt: str = ""
    subtitles: bool = True


class VideoProviderPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video: str = "seedance-2.0"
    image: str = "gpt-image-2"
    music: str = "suno"


class VideoAutomationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["automatic"] = "automatic"
    max_artifact_retries: int = Field(default=1, ge=0, le=3)


def _canonical_video_language(value: str | None, fallback: str = "zh-CN") -> str:
    normalized = (value or "").strip().lower().replace("_", "-")
    if normalized.startswith("zh"):
        return "zh-CN"
    if normalized.startswith("en"):
        return "en-US"
    return fallback


class VideoLanguageContract(BaseModel):
    """Persistent language choices for UI, authored content, speech and subtitles."""

    model_config = ConfigDict(extra="forbid")

    ui_locale: str = Field(default="zh-CN", min_length=2, max_length=20)
    content_language: str = Field(default="zh-CN", min_length=2, max_length=20)
    spoken_language: str = Field(default="zh-CN", min_length=2, max_length=20)
    subtitle_language: str = Field(default="zh-CN", min_length=2, max_length=20)
    provider_prompt_language: str = Field(default="auto", min_length=2, max_length=20)

    @model_validator(mode="after")
    def normalize_supported_languages(self):
        self.ui_locale = _canonical_video_language(self.ui_locale)
        self.content_language = _canonical_video_language(self.content_language)
        self.spoken_language = _canonical_video_language(
            self.spoken_language, self.content_language,
        )
        self.subtitle_language = _canonical_video_language(
            self.subtitle_language, self.content_language,
        )
        self.provider_prompt_language = (
            "auto" if self.provider_prompt_language.strip().lower() == "auto"
            else _canonical_video_language(
                self.provider_prompt_language, self.content_language,
            )
        )
        return self


class _LanguageAwareVideoModel(BaseModel):
    """Compatibility normalization for legacy documents that only stored `language`."""

    @model_validator(mode="after")
    def align_language_contract(self):
        legacy = _canonical_video_language(getattr(self, "language", None))
        contract = getattr(self, "language_contract", None)
        if contract is None:
            contract = VideoLanguageContract(
                ui_locale=legacy,
                content_language=legacy,
                spoken_language=legacy,
                subtitle_language=legacy,
            )
            self.language_contract = contract
        elif legacy != contract.content_language:
            raise ValueError(
                "language must equal language_contract.content_language"
            )
        self.language = contract.content_language
        return self


class ProjectIntent(_LanguageAwareVideoModel):
    """Known project constraints accepted before a complete VideoSpec exists."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    brief: str = Field(min_length=1)
    language: str = Field(default="zh-CN", min_length=2, max_length=20)
    language_contract: VideoLanguageContract | None = None
    target_duration_seconds: float = Field(default=15, gt=0, le=600)
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9"
    resolution: str = Field(default="1080p", pattern=r"^[0-9]{3,4}p$")
    workflow_id: str
    style_id: str = "cuti.cinematic"
    activated_skill_ids: list[str] = Field(default_factory=list)
    source_asset_ids: list[str] = Field(default_factory=list)
    workflow_parameters: dict[str, Any] = Field(default_factory=dict)
    providers: VideoProviderPreferences = Field(default_factory=VideoProviderPreferences)
    automation: VideoAutomationPolicy = Field(default_factory=VideoAutomationPolicy)
    constraints: dict[str, Any] = Field(default_factory=dict)


class VideoSpec(_LanguageAwareVideoModel):
    """Provider-neutral, deterministic input for the first project build."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    language: str = Field(default="zh-CN", min_length=2, max_length=20)
    language_contract: VideoLanguageContract | None = None
    target_duration_seconds: float = Field(gt=0, le=600)
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9"
    resolution: str = Field(default="1080p", pattern=r"^[0-9]{3,4}p$")
    # Workflow selection is an Agent/UI decision. Never silently route an
    # omitted value into the legacy seedance-story compatibility compiler.
    workflow_id: str
    style_id: str = "cuti.cinematic"
    activated_skill_ids: list[str] = Field(default_factory=list)
    source_asset_ids: list[str] = Field(default_factory=list)
    workflow_parameters: dict[str, Any] = Field(default_factory=dict)
    characters: list[VideoCharacterSpec] = Field(default_factory=list)
    shots: list[VideoShotSpec] = Field(min_length=1)
    audio: VideoAudioSpec = Field(default_factory=VideoAudioSpec)
    providers: VideoProviderPreferences = Field(default_factory=VideoProviderPreferences)
    automation: VideoAutomationPolicy = Field(default_factory=VideoAutomationPolicy)

    @model_validator(mode="after")
    def validate_story_graph(self):
        character_ids = [item.id for item in self.characters]
        shot_ids = [item.id for item in self.shots]
        if len(character_ids) != len(set(character_ids)):
            raise ValueError("character ids must be unique")
        if len(shot_ids) != len(set(shot_ids)):
            raise ValueError("shot ids must be unique")
        orders = [item.order for item in self.shots]
        if orders != list(range(1, len(self.shots) + 1)):
            raise ValueError("shots must be ordered consecutively starting at 1")
        known = set(character_ids)
        unknown = sorted({ref for shot in self.shots for ref in shot.character_ids} - known)
        if unknown:
            raise ValueError(f"shots reference unknown characters: {', '.join(unknown)}")
        total = sum(item.duration_seconds for item in self.shots)
        if abs(total - self.target_duration_seconds) > 1:
            raise ValueError("shot durations must equal target duration within one second")
        return self


class RebuildPlanItem(BaseModel):
    superseded_by: str | None = None
    step_id: str = Field(default_factory=uid)
    objective: str = ""
    input_artifact_version_ids: list[str] = Field(default_factory=list)
    artifact_version_id: str = ""
    output_artifact_id: str = ""
    output_artifact_type: str = ""
    action: Literal["create", "rebuild", "validate", "reuse"]
    capability: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    idempotency_key: str = ""
    estimated_cost: float = Field(default=0.0, ge=0)
    order: int | None = None
    # Steps in the same execution group are independent and may run together.
    # This is a durable plan contract rather than an executor-side guess: Cuti
    # workflows differ materially here (for example, short-drama segments are
    # parallel while Seedance continuation shots are deliberately serial).
    execution_group: str | None = None
    max_parallelism: int | None = Field(default=None, ge=1)
    reason: str = ""
    skill_ids: list[str] = Field(default_factory=list)
    resolved_skills: list[ResolvedSkillRef] = Field(default_factory=list)
    skill_context: SkillContext | None = None
    constraint_contract: SkillConstraintContract | None = None
    language_contract: VideoLanguageContract | None = None

    @field_validator("idempotency_key", mode="before")
    @classmethod
    def normalize_legacy_empty_idempotency_key(cls, value):
        return "" if value is None else value


class RebuildPlan(BaseModel):
    id: str = Field(default_factory=uid)
    project_id: str
    kind: Literal["initial", "incremental", "export"] = "incremental"
    change_request_id: str | None = None
    base_project_version_id: str
    workflow_id: str = ""
    video_spec: VideoSpec | None = None
    project_intent: ProjectIntent | None = None
    items: list[RebuildPlanItem]
    schema_version: Literal[1, 2] = 1
    current_revision: int = Field(default=1, ge=1)
    video_spec_revision_id: str | None = None
    current_phase: str = "full"
    next_checkpoint: "PlanCheckpointDefinition | None" = None
    estimated_cost: float = 0.0
    status: Literal["preview", "applied", "superseded"] = "preview"
    created_at: datetime = Field(default_factory=now)

    def ids_for(self, action: Literal["create", "rebuild", "validate", "reuse"]) -> list[str]:
        return [item.artifact_version_id for item in self.items if item.action == action]


# Public name for new callers; RebuildPlan remains as a compatibility name.
BuildPlan = RebuildPlan


class BuildStep(BaseModel):
    id: str = Field(default_factory=uid)
    build_id: str
    project_id: str
    plan_step_id: str
    action: Literal["create", "rebuild", "validate", "reuse"]
    capability: str = ""
    status: Literal[
        "pending", "running", "waiting_external", "completed", "failed", "cancelled"
    ] = "pending"
    attempt: int = Field(default=0, ge=0)
    remote_operation_id: str | None = None
    remote_provider: str | None = None
    result_artifact_version_id: str | None = None
    error: str | None = None
    resolved_skills: list[ResolvedSkillRef] = Field(default_factory=list)
    skill_context: SkillContext | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime = Field(default_factory=now)


class Build(BaseModel):
    id: str = Field(default_factory=uid)
    project_id: str
    plan_id: str
    base_project_version_id: str
    idempotency_key: str
    session_id: str | None = None
    user_id: str | None = None
    kind: Literal["initial", "incremental", "export"] = "incremental"
    status: Literal[
        "queued", "running", "waiting_external", "waiting_agent",
        "completed", "failed", "cancelled",
    ] = "queued"
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = "Queued"
    project_version_id: str | None = None
    estimated_cost: float = Field(default=0.0, ge=0)
    actual_cost: float = Field(default=0.0, ge=0)
    error: str | None = None
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class VideoSpecRevision(BaseModel):
    """Immutable partial or complete VideoSpec state produced at a planning checkpoint."""

    id: str = Field(default_factory=uid)
    project_id: str
    revision: int = Field(ge=1)
    parent_revision_id: str | None = None
    content: dict[str, Any] = Field(default_factory=dict)
    resolved_sections: list[str] = Field(default_factory=list)
    unresolved_sections: list[str] = Field(default_factory=list)
    source_artifact_version_ids: list[str] = Field(default_factory=list)
    checkpoint_id: str | None = None
    created_by: Literal["user", "agent", "migration"] = "agent"
    complete: bool = False
    created_at: datetime = Field(default_factory=now)


class BuildPlanRevision(BaseModel):
    """Append-only record of one accepted change to a durable BuildPlan."""

    id: str = Field(default_factory=uid)
    plan_id: str
    project_id: str
    revision: int = Field(ge=1)
    base_revision: int = Field(ge=0)
    checkpoint_id: str | None = None
    video_spec_revision_id: str | None = None
    added_step_ids: list[str] = Field(default_factory=list)
    cancelled_step_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    created_at: datetime = Field(default_factory=now)


class PlanCheckpointDefinition(BaseModel):
    """Workflow-owned declaration for the next semantic planning boundary."""

    id: str
    phase: str
    next_phase: str
    required_artifact_types: list[str] = Field(default_factory=list)
    resolves: list[str] = Field(default_factory=list)
    planner_instruction: str = ""
    planning_mode: Literal["staged", "agentic"] = "staged"
    max_planning_attempts: int = Field(default=3, ge=1, le=5)


class PlanCheckpoint(BaseModel):
    """Durable request for the bound Harness Session to plan a later build phase."""

    id: str = Field(default_factory=uid)
    project_id: str
    build_id: str
    plan_id: str
    workflow_id: str
    session_id: str
    user_id: str
    phase: str
    next_phase: str
    status: Literal["pending", "planning", "resolved", "failed"] = "pending"
    required_artifact_types: list[str] = Field(default_factory=list)
    artifact_version_ids: list[str] = Field(default_factory=list)
    artifact_summaries: list[dict[str, Any]] = Field(default_factory=list)
    resolved_sections: list[str] = Field(default_factory=list)
    unresolved_sections: list[str] = Field(default_factory=list)
    planner_instruction: str = ""
    planning_mode: Literal["staged", "agentic"] = "staged"
    base_plan_revision: int = Field(ge=1)
    base_spec_revision: int = Field(ge=1)
    delivery_attempts: int = Field(default=0, ge=0)
    max_delivery_attempts: int = Field(default=3, ge=1, le=5)
    semantic_repair_attempts: int = Field(default=0, ge=0, le=1)
    delivery_id: str | None = None
    lease_expires_at: datetime | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class CheckpointResolution(BaseModel):
    """One Cuti-compatible, append-only PlanPatch authored by the Agent."""

    model_config = ConfigDict(extra="forbid")

    base_plan_revision: int = Field(ge=1)
    base_spec_revision: int = Field(ge=1)
    idempotency_key: str = Field(min_length=1, max_length=200)
    video_spec: VideoSpec | None = None
    video_spec_patch: dict[str, Any] | None = None
    phase_inputs: dict[str, Any] = Field(default_factory=dict)
    proposed_steps: list[RebuildPlanItem] = Field(default_factory=list)
    cancel_step_ids: list[str] = Field(default_factory=list)
    replace_failed_step_ids: dict[str, str] = Field(default_factory=dict)
    goal_satisfied: bool = False
    waiting_for_input: bool = False
    response: str = ""
    reason: str = ""

    @model_validator(mode="after")
    def exactly_one_spec_input(self):
        if (self.video_spec is None) == (self.video_spec_patch is None):
            raise ValueError("provide exactly one of video_spec or video_spec_patch")
        return self


class ValidationResult(BaseModel):
    id: str = Field(default_factory=uid)
    project_id: str
    build_id: str
    artifact_version_id: str | None = None
    validator_id: str
    passed: bool
    score: float | None = None
    issues: list[str] = Field(default_factory=list)
    # ``metadata["inconclusive"] = True`` marks a failing result whose check reached
    # no verdict (for example a failed VLM call); Agent feedback does not report it as
    # a rejection of the artifact.
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now)


class TimelineItem(BaseModel):
    id: str = Field(default_factory=uid)
    artifact_version_id: str
    track: str
    start_seconds: float = Field(ge=0)
    duration_seconds: float = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimelineArtifact(BaseModel):
    duration_seconds: float = Field(default=0, ge=0)
    items: list[TimelineItem] = Field(default_factory=list)


class ProjectEvent(BaseModel):
    id: str = Field(default_factory=uid)
    project_id: str
    sequence: int = 0
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now)


class ExportRecord(BaseModel):
    id: str = Field(default_factory=uid)
    project_id: str
    project_version_id: str
    idempotency_key: str
    format: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    uri: str | None = None
    created_at: datetime = Field(default_factory=now)
