from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import tempfile
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import httpx

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .api import _artifact_payload, get_runtime
from .deepseek_client import DeepSeekHarnessClient, DeepSeekHarnessError
from .runtime import VideoBuildRuntime
from .models import MediaArtifactVersion
from .repository import ProjectVersionConflict, SidebarRun
from .upload_security import sign_uploaded_file as _sign_uploaded_file
from .workflow_plans import workflow_execution_kind
from app.chat.utils.file_utils import process_uploaded_files
from app.exceptions import BusinessException, BusinessExceptionCode
from app.chat.v2.language import (
    LanguageSkillConflictError,
    active_language_skill,
    explicit_language_skill,
    normalize_output_language,
    resolve_video_language_contract,
    video_language_instruction,
)


router = APIRouter(prefix="/v2", tags=["deepseek-compatibility-bff"])
studio_router = APIRouter(prefix="/studio", tags=["deepseek-studio-compatibility-bff"])
client: DeepSeekHarnessClient | None = None
logger = logging.getLogger(__name__)
_DSH_IMAGE_MEDIA_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
_SOURCE_IMAGE_ATTACH_NOTE = (
    "Source images attached to this message correspond to those Source Artifacts, "
    "in the same order."
)
_MEDIA_VISIBILITY_INSTRUCTION = (
    "You can see uploaded still images attached to this turn. You cannot see video "
    "pixels or hear audio. To understand an uploaded video, call media.video_analyze "
    "with that Source Artifact url or video_step; the result is a videomap Artifact. "
    "To understand music or audio structure, call media.audio_analyze; the result is "
    "an audiomap Artifact. Do not claim you watched or listened to media you did not analyze."
)


def set_deepseek_client(value: DeepSeekHarnessClient | None) -> None:
    global client
    client = value


def get_deepseek_client() -> DeepSeekHarnessClient:
    if client is None:
        raise HTTPException(status_code=503, detail="DeepSeek compatibility backend is not configured")
    return client


async def identity(
    user_id: str | None = Header(default=None, alias="X-Video-User-Id"),
) -> str:
    return user_id or os.getenv("VIDEO_RUNTIME_LOCAL_USER_ID", "local-user")


class CreateRunBody(BaseModel):
    objective: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    thread_id: str | None = None
    user_option: dict[str, Any] | None = None
    input_files: list[dict[str, Any]] = Field(default_factory=list)
    workflow_id: str | None = None
    activated_skill_ids: list[str] = Field(default_factory=list)
    automatic_video: bool = False
    planning_off: bool | None = None
    code_video_mode: bool | None = None


class MessageBody(BaseModel):
    content: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    thread_id: str | None = None
    resume_builds: bool = False
    user_option: dict[str, Any] | None = None
    input_files: list[dict[str, Any]] = Field(default_factory=list)
    workflow_id: str | None = None
    activated_skill_ids: list[str] = Field(default_factory=list)
    automatic_video: bool = False
    planning_off: bool | None = None
    code_video_mode: bool | None = None


class StudioProjectBody(BaseModel):
    objective: str = Field(min_length=1)
    thread_id: str | None = None
    user_option: dict[str, Any] | None = None
    input_files: list[dict[str, Any]] = Field(default_factory=list)
    workflow_id: str | None = None
    activated_skill_ids: list[str] = Field(default_factory=list)


class StudioSkillEnableBody(BaseModel):
    enabled: bool


def success(data: Any) -> dict[str, Any]:
    return {"code": 0, "message": "success", "data": data}


@router.get("/runtime-options")
async def runtime_options() -> dict[str, Any]:
    """Expose credential presence only, never credential values."""
    wavespeed_configured = _wavespeed_configured()
    return success({
        "wavespeed_configured": wavespeed_configured,
        "default_code_video_mode": not wavespeed_configured,
    })


def _language_overrides(user_option: dict[str, Any] | None) -> dict[str, str]:
    options = user_option or {}
    nested = options.get("language_contract")
    source = nested if isinstance(nested, dict) else options
    return {
        key: str(source[key])
        for key in (
            "ui_locale", "content_language", "spoken_language",
            "subtitle_language", "provider_prompt_language",
        )
        if isinstance(source.get(key), str) and str(source[key]).strip()
    }


def _request_language_contract(
    text: str,
    *,
    app_language: str | None,
    user_option: dict[str, Any] | None,
    activated_skill_ids: list[str] | None = None,
    current: dict[str, str] | None = None,
) -> dict[str, str]:
    try:
        explicit_skill = explicit_language_skill(text)
        selected_skill = explicit_skill or active_language_skill(activated_skill_ids)
        return resolve_video_language_contract(
            text,
            ui_locale=app_language,
            current=current,
            overrides=_language_overrides(user_option),
            language_skill=selected_skill,
        )
    except LanguageSkillConflictError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _prompt_input_files(input_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose project Artifact identities to the model without storage URLs or receipts."""
    return [{
        "artifact_id": item.get("artifact_id"),
        "type": item.get("type"),
        "filename": item.get("filename"),
    } for item in input_files]


def _sniff_image_media_type(raw: bytes, filename: str | None = None) -> str | None:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw[:6] in {b"GIF87a", b"GIF89a"}:
        return "image/gif"
    ext = Path(str(filename or "")).suffix.lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(ext)


async def _load_source_image_bytes(url: str) -> bytes | None:
    """Read source pixels for session.prompt without putting the URL in the text prompt."""
    value = str(url or "").strip()
    if not value:
        return None
    if value.startswith("data:") and ";base64," in value:
        try:
            return base64.b64decode(value.split(";base64,", 1)[1])
        except Exception:
            return None
    path = Path(value)
    if path.is_file():
        return await asyncio.to_thread(path.read_bytes)
    try:
        from app.utils.file_utils import inline_local_image_url_for_llm

        inlined = await inline_local_image_url_for_llm(value)
        if inlined.startswith("data:") and ";base64," in inlined:
            return base64.b64decode(inlined.split(";base64,", 1)[1])
    except Exception as exc:
        logger.warning("source image local inline failed url=%s err=%s", value[:120], exc)
    if value.startswith(("http://", "https://")):
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as http:
                response = await http.get(value)
                response.raise_for_status()
                return response.content
        except Exception as exc:
            logger.warning("source image fetch failed url=%s err=%s", value[:120], exc)
            return None
    return None


async def _source_image_parts(input_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """DSH prompt image blocks for uploaded source stills. Audio/video stay text-only."""
    parts: list[dict[str, Any]] = []
    for item in input_files:
        kind = str(item.get("type") or "").strip().lower()
        if kind != "image":
            continue
        url = str(item.get("url") or item.get("uri") or "").strip()
        raw = await _load_source_image_bytes(url)
        if not raw:
            logger.warning("skipping unread source image artifact_id=%s", item.get("artifact_id"))
            continue
        media_type = _sniff_image_media_type(raw, str(item.get("filename") or url))
        if media_type not in _DSH_IMAGE_MEDIA_TYPES:
            logger.warning(
                "skipping unsupported source image media artifact_id=%s", item.get("artifact_id"),
            )
            continue
        part: dict[str, Any] = {
            "type": "image",
            "mediaType": media_type,
            "data": base64.b64encode(raw).decode("ascii"),
        }
        name = str(item.get("filename") or item.get("artifact_id") or "").strip()
        if name:
            part["name"] = name
        parts.append(part)
    return parts


async def _prompt_with_source_images(
    dsh: DeepSeekHarnessClient,
    session_id: str,
    text: str,
    input_files: list[dict[str, Any]] | None = None,
) -> None:
    images = await _source_image_parts(input_files or [])
    if images:
        text = f"{text}\n{_SOURCE_IMAGE_ATTACH_NOTE}"
    await dsh.prompt(session_id, text, images=images)


async def _import_input_files(
    runtime: VideoBuildRuntime,
    project_id: str,
    user_id: str,
    input_files: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Commit verified uploads as project-owned immutable source artifacts."""
    current = {item.artifact_id: item for item in await runtime.repo.current_artifacts(project_id)}
    imported: list[dict[str, Any]] = []
    for item in input_files:
        kind = str(item.get("type") or "").strip().lower()
        url = str(item.get("url") or "").strip()
        metadata = dict(item.get("metadata") or {})
        receipt = str(metadata.get("upload_receipt") or "")
        if kind not in {"image", "audio", "video"} or not url:
            raise HTTPException(status_code=422, detail="Uploaded media has an invalid type or URL")
        if not hmac.compare_digest(receipt, _sign_uploaded_file(user_id, kind, url)):
            raise HTTPException(status_code=422, detail="Uploaded media receipt is missing or invalid")
        logical_id = str(metadata.get("artifact_id") or f"source:{hashlib.sha256(url.encode()).hexdigest()[:24]}")
        artifact = current.get(logical_id)
        if artifact is None:
            artifact = await runtime.repo.add_artifact(MediaArtifactVersion(
                artifact_id=logical_id,
                project_id=project_id,
                type=f"source_{kind}",
                uri=url,
                title=str(item.get("filename") or f"Uploaded {kind}"),
                summary="Project-owned uploaded source media",
                content_digest=hashlib.sha256(url.encode()).hexdigest(),
                provider_id="cuti-upload",
                provenance={"uploaded_by": user_id},
                metadata={**metadata, "media_type": kind, "source": "upload"},
            ))
            current[logical_id] = artifact
        imported.append({
            **item,
            "artifact_id": logical_id,
            "metadata": {**metadata, "artifact_id": logical_id},
        })
    return imported


async def _project_and_binding(
    runtime: VideoBuildRuntime,
    project_id: str,
    user_id: str,
):
    try:
        project = await runtime.repo.get_project(project_id)
        if project.user_id != user_id:
            raise LookupError("project not found")
        binding = await runtime.repo.latest_session_binding(project_id, user_id)
        return project, binding
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="V2 run not found") from exc


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text", ""))
            for item in value
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _json_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _tool_result(data: dict[str, Any]) -> tuple[str, str, bool]:
    message = data.get("message") if isinstance(data.get("message"), dict) else {}
    source = message.get("source") if isinstance(message.get("source"), dict) else {}
    call_id = str(source.get("callId") or "")
    outputs: list[str] = []
    is_error = False
    for block in message.get("content", []):
        if not isinstance(block, dict) or block.get("type") != "tool-result":
            continue
        call_id = str(block.get("toolCallId") or call_id)
        is_error = is_error or bool(block.get("isError"))
        content = block.get("content")
        if isinstance(content, str):
            outputs.append(content)
        elif isinstance(content, list):
            outputs.extend(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
    return call_id, "\n".join(item for item in outputs if item), is_error


def _usage_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    chunk = data.get("chunk") if isinstance(data.get("chunk"), dict) else {}
    usage = chunk.get("usage") if isinstance(chunk.get("usage"), dict) else None
    if chunk.get("type") != "usage" or usage is None:
        return None
    input_tokens = int(usage.get("inputTokens") or 0)
    output_tokens = int(usage.get("outputTokens") or 0)
    cache_read_tokens = int(usage.get("cacheReadTokens") or 0)
    cache_write_tokens = int(usage.get("cacheWriteTokens") or 0)
    reasoning_tokens = int(usage.get("reasoningTokens") or 0)
    return {
        "scope": "deepseek-harness",
        "turn": data.get("turn"),
        "step": data.get("step"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": (
            input_tokens + output_tokens + cache_read_tokens + cache_write_tokens
        ),
        "status": "succeeded",
    }


_CREATE_PROMPT_MARKER = "CUTI_VIDEO_CREATE_V1"
_AGENT_PROMPT_MARKER = "CUTI_AGENT_CHAT_V1"
_CHECKPOINT_PROMPT_MARKER = "CUTI_VIDEO_CHECKPOINT_V1"
_RUN_CONTEXT_OPERATION = "compat-run-context"
_RUN_CONTEXT_KEY = "initial"
_LIST_RUNS_LIMIT = 50
_LIST_PREVIEW_MAX = 500
# Chat bubbles stay complete. Trace/progress only needs the newest events on open.
_SNAPSHOT_EVENT_LIMIT = 80
_UNSET = object()
_SKILL_SELECTION_SEPARATOR = "\n\nServer-resolved video Skill selection:\n"
_UI_DEFAULTS_SEPARATOR = (
    "\n\nUI defaults (creation controls) and inputs. "
    "User-stated config in the message has higher priority; "
    "keep defaults only for fields the user did not mention:\n"
)
_LEGACY_CONTROLS_SEPARATOR = "\n\nCurrent creation controls and inputs:\n"


_EXPLICIT_SKILL = re.compile(r"(?<![A-Za-z0-9_-])[$/]([A-Za-z0-9][A-Za-z0-9_-]{0,63})")
_PLANNING_OFF_COMMAND = re.compile(r"(?i)(?:^|\s)/planning-off(?:\s|$)")


def _planning_off_requested(
    text: str,
    requested: bool | None,
    *,
    current: bool = False,
) -> bool:
    """Treat the command as an enable shortcut while allowing the UI to disable it."""
    if _PLANNING_OFF_COMMAND.search(text):
        return True
    return current if requested is None else requested


def _wavespeed_configured() -> bool:
    return bool((os.getenv("WAVESPEED_API_KEY") or "").strip())


def _code_video_mode_requested(
    requested: bool | None,
    *,
    current: bool | None = None,
    planning_off: bool = False,
) -> bool:
    """Default new projects to local Code-to-Video when WaveSpeed is unavailable."""
    if requested is not None:
        return requested
    if current is not None:
        return current
    return not planning_off and not _wavespeed_configured()


def _requested_skill_selection(
    runtime: VideoBuildRuntime,
    requested_workflow: str | None,
    requested_activated: list[str],
    text: str,
) -> tuple[str | None, list[str]]:
    """Resolve explicit UI and `$skill` choices from the unified Runtime catalog."""
    explicit = [
        match.group(1) for match in _EXPLICIT_SKILL.finditer(text)
        if runtime.skills.catalog.has(match.group(1))
    ]
    explicit_workflows = []
    helpers = list(requested_activated)
    for skill_id in explicit:
        metadata = runtime.skills.catalog.load(skill_id).metadata
        if (metadata.metadata or {}).get("kind") == "workflow":
            explicit_workflows.append(skill_id)
        elif skill_id not in helpers:
            helpers.append(skill_id)
    if requested_workflow:
        return requested_workflow, helpers
    unique_workflows = list(dict.fromkeys(explicit_workflows))
    if len(unique_workflows) > 1:
        raise HTTPException(status_code=422, detail="Only one Workflow Skill can be selected per request")
    return (unique_workflows[0] if unique_workflows else None), helpers


def _validate_workflow_selection(
    runtime: VideoBuildRuntime,
    workflow: str | None,
) -> None:
    if not workflow:
        return
    if not runtime.skills.catalog.has(workflow):
        # Dedicated Video Plugins may expose a Workflow alias without adding a
        # synthetic SKILL.md. Accept it only
        # when the loaded plugin publishes an explicit, selectable compiler
        # contract; a bare contributions.workflows entry still fails closed.
        views = []
        for loaded in runtime.plugins.loaded:
            describe = getattr(loaded.implementation, "describe_workflows", None)
            if callable(describe):
                views.extend(describe())
        view = next((item for item in views if item.get("id") == workflow), None)
        if view is None:
            raise HTTPException(
                status_code=422,
                detail=f"Workflow Skill or plugin is not installed: {workflow}",
            )
        if not view.get("available") or not view.get("userSelectable") or not view.get("compiler"):
            reason = view.get("unavailableReason") or "no dedicated selectable compiler"
            raise HTTPException(
                status_code=409,
                detail=f"Workflow is unavailable: {workflow}: {reason}",
            )
        return
    metadata = runtime.skills.catalog.load(workflow).metadata
    if (metadata.metadata or {}).get("kind") != "workflow":
        raise HTTPException(status_code=422, detail=f"Skill is not a workflow: {workflow}")
    spec = runtime.skills.workflows.get(workflow)
    try:
        if spec is None:
            raise ValueError("Workflow is not registered")
        kind = workflow_execution_kind(spec)
        if kind == "agent_plan_patch" and not (
            runtime.staged_planning_enabled and runtime.continuous_plan_patch_enabled
        ):
            raise ValueError("Workflow requires continuous PlanPatch planning to be enabled")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=f"Workflow is unavailable: {workflow}: {exc}") from exc


def _visible_user_text(value: str) -> str:
    """Keep BFF orchestration instructions out of the legacy chat transcript."""
    if value.startswith(f"{_CHECKPOINT_PROMPT_MARKER}\n"):
        return ""
    for separator in (_UI_DEFAULTS_SEPARATOR, _LEGACY_CONTROLS_SEPARATOR):
        if separator in value:
            value = value.split(separator, 1)[0]
            break
    if _SKILL_SELECTION_SEPARATOR in value:
        value = value.split(_SKILL_SELECTION_SEPARATOR, 1)[0]
    if not value.startswith((f"{_CREATE_PROMPT_MARKER}\n", f"{_AGENT_PROMPT_MARKER}\n")):
        return value
    first_line = value.splitlines()[1] if len(value.splitlines()) > 1 else ""
    prefix = "VISIBLE_USER_REQUEST_JSON: "
    if not first_line.startswith(prefix):
        return value
    try:
        parsed = json.loads(first_line[len(prefix):])
    except json.JSONDecodeError:
        return value
    return parsed if isinstance(parsed, str) else value


def _initial_agent_prompt(
    *,
    objective: str,
    project_id: str,
    base_project_version_id: str,
    idempotency_key: str,
    user_option: dict[str, Any] | None,
    input_files: list[dict[str, Any]],
    workflow_id: str | None = None,
    activated_skill_ids: list[str] | None = None,
    language_contract: dict[str, str] | None = None,
) -> str:
    """Give the agent project context without assuming that every message requests a video."""
    options = user_option or {}
    activated = list(dict.fromkeys(activated_skill_ids or []))
    safe_inputs = _prompt_input_files(input_files)
    lines = [
        _AGENT_PROMPT_MARKER,
        f"VISIBLE_USER_REQUEST_JSON: {json.dumps(objective, ensure_ascii=False)}",
        (
            "First interpret the visible user request in the prior conversation context and "
            "decide whether the user is asking for video "
            "project work. Greetings, general conversation, explanations, and questions that do "
            "not request video creation, editing, inspection, or export must receive a normal "
            "direct answer without calling any video_* tool. Do not invent a video task."
        ),
        (
            "The Video Runtime BFF has created an empty draft project only as conversation "
            f"storage and bound project {project_id} to this Session. Its current "
            f"base_project_version_id is {base_project_version_id}. Do not call "
            "video_project_create for this draft."
        ),
        (
            "If and only if the request in conversation context asks for video project work, use the "
            "available video_* tools to satisfy it. Load the explicitly selected Workflow when "
            "present; otherwise list and choose an available Workflow. Load its required Skills, "
            "but do not use media.hyperframes_render unless Code-to-Video mode is explicitly enabled. "
            "inspect the PlanPatch capabilities, plan against the existing project and base "
            f"version with idempotency_key plan:{idempotency_key}, then start the build with "
            f"idempotency_key build:{idempotency_key}."
        ),
        (
            "UI creation controls are optional defaults and apply only if video work is requested: "
            f"{json.dumps(options, ensure_ascii=False, sort_keys=True)}"
        ),
        f"Uploaded project Source Artifacts: {json.dumps(safe_inputs, ensure_ascii=False, sort_keys=True)}",
        _MEDIA_VISIBILITY_INSTRUCTION,
        video_language_instruction(language_contract or resolve_video_language_contract(objective)),
        f"Activated helper Skill ids: {json.dumps(activated, ensure_ascii=False)}.",
    ]
    if workflow_id:
        lines.append(
            f"The project has selected Workflow {json.dumps(workflow_id)}. For requested video "
            "work, load that Workflow and use its exact id. A saved selection alone is not "
            "a new instruction to generate a video."
        )
    lines.append(f"Visible user request: {objective}")
    return "\n".join(lines)


def _planning_off_prompt(
    *,
    objective: str,
    project_id: str,
    base_project_version_id: str,
    idempotency_key: str,
    user_option: dict[str, Any] | None,
    input_files: list[dict[str, Any]],
    language_contract: dict[str, str] | None = None,
) -> str:
    """Minimal, workflow-free routing contract for `/planning-off` sessions."""
    lines = [
        _AGENT_PROMPT_MARKER,
        f"VISIBLE_USER_REQUEST_JSON: {json.dumps(objective, ensure_ascii=False)}",
        "PLANNING-OFF MODE IS ENABLED FOR THIS SESSION.",
        (
            "Do not list, load, select, or inject any Workflow or Skill. Do not call any "
            "video_workflow_* or video_skill_* tool, and do not add workflow_id or "
            "activated_skill_ids."
        ),
        (
            "Keep this same Harness Session, thread_id, project, and its persisted Artifacts across "
            "turns, but plan ONLY the operation explicitly requested in the current user turn. "
            "Classify that turn as ordinary chat, text creation, standalone music/audio creation, "
            "single-image creation, single-video creation, or an explicit media composition/edit. "
            "A request for music means one music Artifact, never a music-video Workflow or a video. "
            "A request for an image means one image Artifact, never a storyboard or video. Do not "
            "invent or expand plot, characters, scenes, shots, research, narrative beats, or future "
            "production requirements that the user did not explicitly request. Preserve the user's "
            "wording and intent, applying only minimum provider normalization."
        ),
        (
            "For ordinary chat, answer directly and call no video_* tool. For text creation, author "
            "the requested final text yourself, then create the smallest PlanPatch containing one "
            "runtime.artifact.persist step. For music creation, use exactly one "
            "atomic.music.generate step. If the user requests a song but did not supply actual "
            "structured lyric text, set has_lyrics=false and auto_lyrics=true; the prompt is then "
            "a music description and must never be sung verbatim. Set has_lyrics=true and "
            "auto_lyrics=false only when prompt contains the actual complete lyrics with sections "
            "such as [Verse] and [Chorus]. Never set both flags true. For image creation, use "
            "exactly one atomic.image.generate "
            "step. For video creation, use exactly one atomic.video.generate step (or "
            "api.provider.generate only when an explicitly named model requires it). Do not call "
            "video_project_plan or video_project_build in this mode. Do not split a request into "
            "multiple shots and do not assemble, mix, or concatenate media unless the current turn "
            "explicitly asks for that operation."
        ),
        (
            "For a later turn that refers to earlier results, call video_artifact_list and select "
            "the real current ArtifactVersion ids from this same project; never regenerate or "
            "replace those inputs merely to make them available. Pass prior images with the images "
            "input role and prior music/audio with the audio role when generating a video. To put "
            "existing music onto an existing video, use one media.mix_audio PlanPatch step with its "
            "video and audio input roles. Use media.concat only when the current turn explicitly "
            "asks to concatenate multiple videos; when both concatenation and audio mixing are "
            "explicitly requested, use only those two necessary steps in dependency order."
        ),
        (
            "For an explicit music duration, pass target_duration to atomic.music.generate. After "
            "generation, inspect the real Artifact duration. If it differs materially from the "
            "requested duration, append one media.audio_cut step using that music Artifact so the "
            "selected final audio has the requested duration; do not claim success merely because "
            "one provider candidate was closer than another."
        ),
        (
            "You may inspect video_plan_patch_capability_list, but do not load any recommended "
            "Skill. Preview the minimal patch with video_plan_patch_preview, apply it with "
            "video_rebuild_apply without asking for another confirmation, and poll "
            "video_build_status until it reaches a terminal state before treating the turn as done."
        ),
        (
            f"The Runtime already created and bound project {project_id}. Do not create another "
            f"project. Use base_project_version_id {base_project_version_id}, idempotency key "
            f"plan:{idempotency_key} for preview and build:{idempotency_key} for apply."
        ),
        "Optional UI defaults; ignore fields irrelevant to the request: "
        + json.dumps(user_option or {}, ensure_ascii=False, sort_keys=True),
        "Uploaded project Source Artifacts: "
        + json.dumps(_prompt_input_files(input_files), ensure_ascii=False, sort_keys=True),
        video_language_instruction(language_contract or resolve_video_language_contract(objective)),
        f"Visible user request: {objective}",
    ]
    return "\n".join(lines)


def _code_video_prompt(
    *,
    objective: str,
    project_id: str,
    base_project_version_id: str,
    idempotency_key: str,
    user_option: dict[str, Any] | None,
    input_files: list[dict[str, Any]],
    language_contract: dict[str, str] | None = None,
) -> str:
    """Route explicit Code-to-Video turns through the local HyperFrames renderer."""
    options = user_option or {}
    return "\n".join([
        _AGENT_PROMPT_MARKER,
        f"VISIBLE_USER_REQUEST_JSON: {json.dumps(objective, ensure_ascii=False)}",
        "CODE-TO-VIDEO MODE IS ENABLED FOR THIS SESSION.",
        (
            "This is an opt-in local rendering mode. For ordinary chat or a request that does not "
            "ask for video creation or revision, answer directly and call no video_* tool. For a "
            "video request, do not call atomic.video.generate, api.provider.generate, WaveSpeed, "
            "or any cloud video-generation capability."
        ),
        (
            "Load hyperframes-core before authoring. Produce one complete, deterministic, "
            "self-contained HTML composition using inline CSS/SVG and the local "
            "assets/gsap.min.js script only. Do not use remote URLs, fetch, XMLHttpRequest, "
            "WebSocket, EventSource, iframe, object, embed, runtime clocks, or unseeded random "
            "values. The root dimensions, data-duration, data-fps, and the paused "
            "window.__timelines timeline must agree with the requested output. Every non-generic "
            "font-family must have a matching @font-face declaration; prefer system-ui, Arial, "
            "sans-serif, serif, or monospace when a specific font is unnecessary."
        ),
        (
            "Create the smallest PlanPatch with exactly one media.hyperframes_render operation. "
            "Use the stable operation step_id code-video so later turns create a new version of "
            "the same logical Artifact. Put the complete HTML in composition_html and pass width, "
            "height, duration, and fps. Preview with video_plan_patch_preview, apply with "
            "video_rebuild_apply without another confirmation, then poll video_build_status to a "
            "terminal state."
        ),
        (
            "For a later revision, inspect the selected code-video Artifact and its "
            "generation_parameters.composition_html, preserve unrelated content and timing, apply "
            "only the requested change, and render the revised complete HTML in the same project."
        ),
        (
            f"The Runtime already created and bound project {project_id}. Use "
            f"base_project_version_id {base_project_version_id}, idempotency key "
            f"plan:{idempotency_key} for preview and build:{idempotency_key} for apply."
        ),
        "UI creation controls: " + json.dumps(options, ensure_ascii=False, sort_keys=True),
        "Uploaded project Source Artifacts (not automatically available inside HTML): "
        + json.dumps(_prompt_input_files(input_files), ensure_ascii=False, sort_keys=True),
        video_language_instruction(language_contract or resolve_video_language_contract(objective)),
        f"Visible user request: {objective}",
    ])


def _initial_video_build_prompt(
    *,
    objective: str,
    project_id: str,
    base_project_version_id: str,
    idempotency_key: str,
    user_option: dict[str, Any] | None,
    input_files: list[dict[str, Any]],
    workflow_id: str | None = None,
    activated_skill_ids: list[str] | None = None,
    language_contract: dict[str, str] | None = None,
) -> str:
    """Give DeepSeek a deterministic product-flow contract for `/create`."""
    options = user_option or {}
    activated = list(dict.fromkeys(activated_skill_ids or []))
    safe_inputs = _prompt_input_files(input_files)
    staged = os.getenv("VIDEO_STAGED_PLANNING_ENABLED", "false").lower() in {
        "1", "true", "yes", "on",
    }
    lines = [
        _CREATE_PROMPT_MARKER,
        f"VISIBLE_USER_REQUEST_JSON: {json.dumps(objective, ensure_ascii=False)}",
        "This request comes from the fully automatic video creation entry.",
        (
            "Planning uses Cuti continuous PlanPatch semantics: start from ProjectIntent only. "
            "The Runtime will persist it, then repeatedly wake this same DeepSeek Session after "
            "each task completion or failure so you can inspect real Artifacts and append or "
            "cancel pending tasks while other work continues. For a user edit during execution, "
            "inspect checkpoint_id=live and submit a PlanPatch with its returned id and revisions. "
            "Do not try to precompile the complete production DAG in this turn."
            if os.getenv("VIDEO_CONTINUOUS_PLAN_PATCH_ENABLED", "false").lower()
            in {"1", "true", "yes", "on"} else
            "Planning uses the configured Workflow planning contract."
        ),
        (
            "The Video Runtime BFF has already completed the video_project_create stage "
            f"and bound project {project_id} to this Session. Do not create another project."
        ),
        f"The required base_project_version_id is {base_project_version_id}.",
        (
            "UI defaults (creation controls): "
            f"{json.dumps(options, ensure_ascii=False, sort_keys=True)}"
        ),
        f"Uploaded project Source Artifacts: {json.dumps(safe_inputs, ensure_ascii=False, sort_keys=True)}",
        _MEDIA_VISIBILITY_INSTRUCTION,
        video_language_instruction(language_contract or resolve_video_language_contract(objective)),
        (
            "Set the legacy language field to language_contract.content_language and set "
            "language_contract to exactly this persisted five-field value. Do not infer a "
            "different language from SKILL.md or provider-prompt instructions."
        ),
        (
            "Use only the uploaded artifact_id values as VideoSpec.source_asset_ids and shot "
            "reference_asset_ids. Do not copy media URLs into VideoSpec and never invent an asset id."
        ),
        (
            "Write ProjectIntent.brief as the working plan: the user request, plus what you can "
            "see in any attached source media. Shots, captions, and timing wait for media that "
            "does not exist yet."
            if staged else
            "Turn the visible user request and creation controls into one complete, valid VideoSpec."
        ),
        (
            "For planning documents that you can already derive from the user request, Skill, "
            "conversation, and completed Artifacts, write their final content in the PlanPatch and "
            "persist it with runtime.artifact.persist. Do not call atomic.text.generate merely to "
            "ask a second LLM to repeat planning you already performed. Reserve atomic.text.generate "
            "for genuinely independent text transformation or generation that is not already known."
        ),
        (
            "For every video-generation task, parameters.prompt is the exact final provider "
            "prompt, not a short story summary. Make it duration-aware and retain all relevant "
            "identity/product, wardrobe, setting, lighting, camera, motion, performance, audio, "
            "continuity, style, exclusion, and user constraints. On later clips and repairs, "
            "inspect earlier task prompts and Artifact generation_context and preserve comparable "
            "specificity. Adapt temporal beats to the requested duration; do not assume a fixed "
            "number or duration of clips."
        ),
        (
            f"Set {'ProjectIntent' if staged else 'VideoSpec'}.activated_skill_ids exactly to "
            f"{json.dumps(activated, ensure_ascii=False)}. "
            "Always use automation.mode automatic."
        ),
        (
            "Configuration the user stated in the visible request has highest priority. "
            "UI defaults apply only to fields the user did not mention. Do not treat UI "
            "defaults as constraints that override the request. Do not infer a control "
            "from creative content that did not name a setting."
        ),
        (
            "Normalize UI provider names for VideoSpec: seedance_2_* means providers.video "
            "seedance-2.0; gpt_image_2 means providers.image gpt-image-2. If the UI value is "
            "auto, follow the loaded Workflow's default. An exact model named by the visible "
            "user request has higher priority: in particular, never normalize Seedance 2.5 "
            "to Seedance 2.0. Use providers.music suno unless the user explicitly requests "
            "another installed provider."
        ),
        "Optional string fields must be omitted or set to an empty string; never send JSON null.",
        (
            "After video_workflow_load, follow that Skill's instructions, including any "
            "video_skill_load and video_skill_read_resource calls it names. Also call "
            "video_skill_load once for every returned skillDependencies entry. Read markdown "
            "links to bundled files and required paths under references/. Helper Skills stay in context; "
            "do not copy them into activated_skill_ids unless the user or a "
            "project lock already activated them. Then call video_plan_patch_capability_list "
            "before proposing tasks. Copy each capability's parameters_schema; do not invent keys."
        ),
    ]
    if workflow_id:
        lines.append(
            f"The workflow is explicitly selected or project-locked: {json.dumps(workflow_id)}. "
            "Call video_workflow_load for it, then set workflow_id exactly to that id."
        )
    else:
        lines.append(
            "No workflow was explicitly selected. Call video_workflow_list, choose the best "
            "available user-selectable workflow from its descriptions, call video_workflow_load "
            "for that workflow, and set workflow_id to its exact id. Never choose an "
            "unavailable workflow and never silently substitute another workflow after a failure."
        )
    if activated:
        lines.append(
            "Call video_skill_load for every activated_skill_id before constructing VideoSpec."
        )
    lines.extend([
        "Then perform these tool calls in order without asking for confirmation:",
        (
            "1. video_project_plan with the existing project_id and base_project_version_id, "
            f"using idempotency_key plan:{idempotency_key} and "
            + ("project_intent." if staged else "video_spec.")
        ),
        (
            "2. video_project_build with the returned plan_id and the same base version, "
            f"using idempotency_key build:{idempotency_key}."
        ),
        "Do not stop after explaining or displaying the plan. Start the build in this turn.",
        f"Visible user request: {objective}",
    ])
    return "\n".join(lines)


async def _effective_skill_selection(
    runtime: VideoBuildRuntime,
    project_id: str,
    requested_workflow: str | None,
    requested_activated: list[str],
) -> tuple[str | None, list[str]]:
    locked_workflow: str | None = None
    activated: list[str] = []
    for lock in await runtime.list_project_skill_locks(project_id):
        if not lock.enabled:
            continue
        if not runtime.skills.catalog.has(lock.skill_id):
            raise HTTPException(status_code=409, detail=f"Locked Skill is no longer installed: {lock.skill_id}")
        metadata = runtime.skills.catalog.load(lock.skill_id).metadata
        if (metadata.metadata or {}).get("kind") == "workflow":
            locked_workflow = lock.skill_id
        else:
            activated.append(lock.skill_id)

    workflow = requested_workflow or locked_workflow
    _validate_workflow_selection(runtime, workflow)
    for skill_id in requested_activated:
        if not runtime.skills.catalog.has(skill_id):
            raise HTTPException(status_code=422, detail=f"Skill is not installed: {skill_id}")
        metadata = runtime.skills.catalog.load(skill_id).metadata
        if (metadata.metadata or {}).get("kind") == "workflow":
            raise HTTPException(status_code=422, detail=f"Workflow Skill must use workflow_id: {skill_id}")
        activated.append(skill_id)
    return workflow, list(dict.fromkeys(activated))


def _selection_context(workflow_id: str | None, activated_skill_ids: list[str]) -> str:
    return _SKILL_SELECTION_SEPARATOR + json.dumps({
        "workflow_id": workflow_id,
        "activated_skill_ids": activated_skill_ids,
        "instruction": (
            "Preserve non-null workflow_id and activated_skill_ids when editing the creative "
            "VideoSpec. Post-production on existing Artifacts is workflow-independent: call "
            "video_artifact_list and video_plan_patch_capability_list, load the recommended Skill, "
            "then use video_plan_patch_preview without switching or recompiling the generation "
            "Workflow. If workflow_id is null, inspect the existing project instead of changing it. "
            "A missing or partial VideoSpec never makes existing artifacts uneditable. "
            "For parameter corrections, submit regenerate_artifact edits with the selected version "
            "IDs and parameter patches; saved recipes preserve the other parameters. "
            "For an active continuous build, inspect its live checkpoint to append tasks or cancel "
            "pending tasks. For a completed build requiring new creative work, plan a new "
            "ProjectIntent in the SAME project at its current version; selected artifacts are "
            "retained as reusable inputs. Keep completed tasks immutable."
        ),
    }, ensure_ascii=False, sort_keys=True)


def _event_entries(history: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in history.get("events", []):
        if isinstance(entry, dict) and isinstance(entry.get("event"), dict):
            result.append(entry["event"])
    return result


def _status(events: list[dict[str, Any]]) -> str:
    if not events:
        return "planning"
    boundary = next(
        (item for item in reversed(events) if item.get("type") in {"turn/start", "turn/end"}),
        None,
    )
    if boundary is None or boundary.get("type") == "turn/start":
        return "running"
    reason = boundary.get("data", {}).get("reason", {}) if isinstance(boundary.get("data"), dict) else {}
    kind = reason.get("kind") if isinstance(reason, dict) else None
    if kind == "error":
        return "failed"
    if kind == "aborted":
        return "cancelled"
    return "completed"


def _status_from_build(status: str | None) -> str:
    return {
        "queued": "running",
        "running": "running",
        "waiting_agent": "running",
        "waiting_external": "waiting_external",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
    }.get(status or "", "planning")


def _decode_run_context(encoded: str | None) -> dict[str, Any]:
    if not encoded:
        return {}
    try:
        parsed = json.loads(encoded)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _list_preview(result: dict[str, Any]) -> dict[str, str]:
    return {
        "updated_at": str(result.get("updated_at") or ""),
        "last_response": str(result.get("last_response") or "")[:_LIST_PREVIEW_MAX],
        "status": str(result.get("status") or "planning"),
        "objective": str(result.get("objective") or "")[:_LIST_PREVIEW_MAX],
    }


async def _persist_list_preview(
    runtime: VideoBuildRuntime,
    project_id: str,
    context: dict[str, Any],
    result: dict[str, Any],
) -> None:
    preview = _list_preview(result)
    if context.get("list_preview") == preview:
        return
    updated = {**context, "list_preview": preview}
    await runtime.repo.replace_operation_result(
        project_id,
        _RUN_CONTEXT_OPERATION,
        _RUN_CONTEXT_KEY,
        json.dumps(updated, ensure_ascii=False, sort_keys=True),
    )


def _run_from_sidebar(row: SidebarRun) -> dict[str, Any]:
    context = _decode_run_context(row.context_json)
    preview = context.get("list_preview") if isinstance(context.get("list_preview"), dict) else {}
    language_values = context.get("language_contract") if isinstance(context.get("language_contract"), dict) else {}
    output_language = normalize_output_language(
        str(language_values.get("content_language") or "")
    ) or "en"
    status = str(preview.get("status") or "") or _status_from_build(row.latest_build_status)
    return {
        "id": row.project.id,
        "thread_id": row.session_id,
        "project_id": row.project.id,
        "title": row.project.title,
        "objective": str(preview.get("objective") or row.project.title),
        "status": status,
        "current_revision": 0,
        "last_response": str(preview.get("last_response") or ""),
        "output_language": output_language,
        "language_contract": language_values or None,
        "user_option": context.get("user_option"),
        "input_files": context.get("input_files", []),
        "workflow_id": context.get("workflow_id"),
        "activated_skill_ids": context.get("activated_skill_ids", []),
        "automatic_video": bool(context.get("automatic_video", False)),
        "planning_off": bool(context.get("planning_off", False)),
        "code_video_mode": bool(context.get("code_video_mode", False)),
        "skill_locks": [],
        "created_at": row.project.created_at.isoformat(),
        "updated_at": str(preview.get("updated_at") or row.project.updated_at.isoformat()),
    }


def _messages(project_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("type")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if event_type == "user/message":
            source = data.get("source")
            if isinstance(source, dict) and source.get("kind") != "user":
                continue
            role, content = "user", _visible_user_text(_text(data.get("content")))
        elif event_type == "assistant/message":
            message = data.get("message") if isinstance(data.get("message"), dict) else {}
            role, content = "assistant", _text(message.get("content"))
        else:
            continue
        if not content:
            continue
        milliseconds = event.get("time") if isinstance(event.get("time"), (int, float)) else 0
        created = datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()
        messages.append({
            "id": f"dsh-{project_id}-{event.get('seq', len(messages))}",
            "run_id": project_id,
            "role": role,
            "content": content,
            "sequence": event.get("seq", len(messages) + 1),
            "metadata": {"source": "deepseek-harness"},
            "created_at": created,
        })
    return messages


def _run(
    project,
    session_id: str,
    events: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    messages = _messages(project.id, events)
    assistant = [item["content"] for item in messages if item["role"] == "assistant"]
    user = [item["content"] for item in messages if item["role"] == "user"]
    language_values = (context or {}).get("language_contract")
    language_values = language_values if isinstance(language_values, dict) else {}
    output_language = normalize_output_language(
        str(language_values.get("content_language") or "")
    ) or "en"
    return {
        "id": project.id,
        "thread_id": session_id,
        "project_id": project.id,
        "title": project.title,
        "objective": user[0] if user else project.title,
        "status": _status(events),
        "current_revision": 0,
        "last_response": assistant[-1] if assistant else "",
        "output_language": output_language,
        "language_contract": language_values or None,
        "user_option": (context or {}).get("user_option"),
        "input_files": (context or {}).get("input_files", []),
        "workflow_id": (context or {}).get("workflow_id"),
        "activated_skill_ids": (context or {}).get("activated_skill_ids", []),
        "automatic_video": bool((context or {}).get("automatic_video", False)),
        "planning_off": bool((context or {}).get("planning_off", False)),
        "code_video_mode": bool((context or {}).get("code_video_mode", False)),
        "skill_locks": [],
        "created_at": project.created_at.isoformat(),
        "updated_at": max(
            [project.updated_at, *[
                datetime.fromtimestamp(item["time"] / 1000, timezone.utc)
                for item in events if isinstance(item.get("time"), (int, float))
            ]]
        ).isoformat(),
    }


async def _run_with_context(
    runtime: VideoBuildRuntime,
    project,
    session_id: str,
    events: list[dict[str, Any]],
    *,
    persist_preview: bool = True,
    latest_build_status: Any = _UNSET,
) -> dict[str, Any]:
    context = await _load_run_context(runtime, project.id)
    result = _run(project, session_id, events, context)
    locks = await runtime.list_project_skill_locks(project.id)
    result["skill_locks"] = [item.model_dump(mode="json") for item in locks]
    locked_workflow = next((
        item.skill_id for item in locks
        if item.enabled
        and runtime.skills.catalog.has(item.skill_id)
        and (runtime.skills.catalog.load(item.skill_id).metadata.metadata or {}).get("kind") == "workflow"
    ), None)
    if locked_workflow and not context.get("planning_off"):
        result["workflow_id"] = locked_workflow
    if latest_build_status is _UNSET:
        builds = await runtime.repo.list_builds(project.id)
        latest_build_status = (
            max(builds, key=lambda item: item.created_at).status if builds else None
        )
    if latest_build_status == "cancelled":
        result["status"] = "cancelled"
    if persist_preview:
        await _persist_list_preview(runtime, project.id, context, result)
    return result


async def _load_run_context(
    runtime: VideoBuildRuntime,
    project_id: str,
) -> dict[str, Any]:
    encoded = await runtime.repo.get_operation_result(
        project_id, _RUN_CONTEXT_OPERATION, _RUN_CONTEXT_KEY,
    )
    context: dict[str, Any] = {}
    if encoded:
        try:
            parsed = json.loads(encoded)
            if isinstance(parsed, dict):
                context = parsed
        except json.JSONDecodeError:
            pass
    if context.get("language_contract_version") != 1:
        builds = await runtime.repo.list_builds(project_id)
        for build in sorted(builds, key=lambda item: item.created_at, reverse=True):
            try:
                plan = await runtime.repo.get_plan(build.plan_id)
            except LookupError:
                continue
            language_source = plan.video_spec or plan.project_intent
            if language_source is None or language_source.language_contract is None:
                continue
            persisted = language_source.language_contract.model_dump(mode="json")
            source_values = language_source.model_dump(mode="json")
            workflow_values = source_values.get("workflow_parameters")
            constraints = source_values.get("constraints")
            legacy_overrides = {"content_language": persisted["content_language"]}
            for values in (workflow_values, constraints):
                if not isinstance(values, dict):
                    continue
                spoken = values.get("spoken_language") or values.get("dialogue_language")
                subtitles = values.get("subtitle_language")
                provider_prompt = values.get("provider_prompt_language")
                if isinstance(spoken, str) and spoken.strip():
                    legacy_overrides["spoken_language"] = spoken
                if isinstance(subtitles, str) and subtitles.strip():
                    legacy_overrides["subtitle_language"] = subtitles
                if isinstance(provider_prompt, str) and provider_prompt.strip():
                    legacy_overrides["provider_prompt_language"] = provider_prompt
            context["language_contract"] = resolve_video_language_contract(
                str(source_values.get("brief") or source_values.get("title") or ""),
                ui_locale=persisted["ui_locale"],
                overrides=legacy_overrides,
            )
            context["language_contract_version"] = 1
            await runtime.repo.remember_operation_result(
                project_id,
                _RUN_CONTEXT_OPERATION,
                _RUN_CONTEXT_KEY,
                json.dumps(context, ensure_ascii=False, sort_keys=True),
            )
            break
        else:
            context["language_contract_version"] = 1
            if isinstance(context.get("language_contract"), dict):
                await runtime.repo.remember_operation_result(
                    project_id,
                    _RUN_CONTEXT_OPERATION,
                    _RUN_CONTEXT_KEY,
                    json.dumps(context, ensure_ascii=False, sort_keys=True),
                )
    return context


async def _history(dsh: DeepSeekHarnessClient, session_id: str) -> list[dict[str, Any]]:
    try:
        return _event_entries(await dsh.history(session_id, max_messages=1000))
    except DeepSeekHarnessError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


async def _create_route_session(
    dsh: DeepSeekHarnessClient,
    requested_session_id: str | None,
) -> str:
    """Create the route Session, replacing only an orphaned id collision.

    A local Runtime snapshot can be restored from an older backup while the
    Harness Session store still contains the URL's Session. That URL has no
    authoritative Project binding, so keeping its id would mix two project
    histories. Start a fresh Session and let the returned ``thread_id`` move
    Create Space to the new durable binding.
    """
    preset = os.getenv("VIDEO_AGENT_PRESET", "video")
    if requested_session_id:
        existing_session_ids = {
            str(item.get("sessionId"))
            for item in await dsh.list_sessions()
            if item.get("sessionId")
        }
        if requested_session_id in existing_session_ids:
            return await dsh.create_session(agent_preset=preset)
    return await dsh.create_session(
        session_id=requested_session_id,
        agent_preset=preset,
    )


async def _snapshot(
    runtime: VideoBuildRuntime,
    dsh: DeepSeekHarnessClient,
    project,
    session_id: str,
) -> dict[str, Any]:
    project = await runtime.repo.get_project(project.id)
    events, artifacts, version, builds = await asyncio.gather(
        _history(dsh, session_id),
        runtime.repo.list_artifact_versions(project.id),
        runtime.repo.get_project_version(project.current_version_id),
        runtime.repo.list_builds(project.id),
    )
    messages = _messages(project.id, events)
    runtime_tasks: list[dict[str, Any]] = []
    status_map = {
        "pending": "ready", "completed": "succeeded",
        "running": "running", "waiting_external": "waiting_external",
        "failed": "failed", "cancelled": "cancelled",
    }
    steps_by_build = await runtime.repo.list_build_steps_for_builds(
        project.id, [item.id for item in builds[:3]],
    )
    for build in builds[:3]:
        from .moderation_recovery import recovery_message
        from .validation_feedback import task_failure_details, validation_rejections
        plan = await runtime.repo.get_plan(build.plan_id)
        planned = {item.step_id: item for item in plan.items}
        states = {item.plan_step_id: item for item in steps_by_build.get(build.id, [])}
        rejections = await validation_rejections(runtime, project.id, plan, states)
        for step in steps_by_build.get(build.id, []):
            item = planned.get(step.plan_step_id)
            recovery = recovery_message(plan, item, step, states) if item else None
            runtime_tasks.append({
                "id": step.id,
                "run_id": project.id,
                "revision": 0,
                "capability_id": step.capability or step.plan_step_id,
                "objective": step.plan_step_id.replace("-", " "),
                "input_artifact_version_ids": [],
                "depends_on": [],
                "status": (
                    "cancelled" if build.status == "cancelled"
                    and step.status in {"pending", "running", "waiting_external"}
                    else status_map.get(step.status, "ready")
                ),
                "remote_operation_id": step.remote_operation_id,
                "error": step.error,
                "failure": (
                    task_failure_details(plan, item, step, states, rejections)
                    if item else None
                ),
                "progress": 100 if step.status == "completed" else round(build.progress * 100),
                "progress_message": recovery or build.message,
                "created_at": build.created_at.isoformat(),
                "updated_at": step.updated_at.isoformat(),
                "resolved_skills": [
                    item.model_dump(mode="json") for item in step.resolved_skills
                ],
            })
    latest_build_status = builds[0].status if builds else None
    mapped_events = [
        mapped for item in events
        if (mapped := _compat_event(project.id, item)) is not None
    ]
    has_more_events = len(mapped_events) > _SNAPSHOT_EVENT_LIMIT
    return {
        "run": await _run_with_context(
            runtime, project, session_id, events,
            latest_build_status=latest_build_status,
        ),
        "revisions": [],
        "tasks": runtime_tasks,
        "artifacts": [{
            **_artifact_payload(project.id, item, set(version.selections.values())),
            "artifact_type": item.type,
            "produced_by_task_id": item.provenance.get("task_id", ""),
        } for item in artifacts],
        "selections": [{
            "project_id": project.id,
            "type": item.type,
            "artifact_type": item.type,
            "artifact_version_id": item.id,
            "updated_at": project.updated_at.isoformat(),
        } for item in artifacts if version.selections.get(item.artifact_id) == item.id],
        "messages": messages,
        "events": mapped_events[-_SNAPSHOT_EVENT_LIMIT:],
        "has_more_events": has_more_events,
        "last_event_sequence": max((item.get("seq", -1) + 1 for item in events), default=0),
    }


@router.post("/runs")
async def create_run(
    body: CreateRunBody,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
    dsh: Annotated[DeepSeekHarnessClient, Depends(get_deepseek_client)],
    app_language: Annotated[str | None, Header(alias="X-App-Language")] = None,
) -> dict[str, Any]:
    existing = await runtime.repo.get_compatibility_run(user_id, body.idempotency_key)
    if existing is not None:
        project_id, session_id = existing
        project, _binding = await _project_and_binding(runtime, project_id, user_id)
        return success(await _run_with_context(
            runtime, project, session_id, await _history(dsh, session_id),
        ))
    planning_off = _planning_off_requested(body.objective, body.planning_off)
    code_video_mode = _code_video_mode_requested(
        body.code_video_mode,
        planning_off=planning_off,
    )
    if planning_off and code_video_mode:
        raise HTTPException(
            status_code=422,
            detail="planning-off and Code-to-Video mode cannot be enabled together",
        )
    if planning_off or code_video_mode:
        requested_workflow, requested_activated = None, []
    else:
        requested_workflow, requested_activated = _requested_skill_selection(
            runtime, body.workflow_id, body.activated_skill_ids, body.objective,
        )
        _validate_workflow_selection(runtime, requested_workflow)
    try:
        session_id = await _create_route_session(dsh, body.thread_id)
        project, initial_version = await runtime.create_project(
            user_id=user_id,
            title=body.objective[:200],
        )
        await runtime.bind_session(project_id=project.id, session_id=session_id, user_id=user_id)
        imported_input_files = await _import_input_files(
            runtime, project.id, user_id, body.input_files,
        )
        if imported_input_files:
            project = await runtime.repo.get_project(project.id)
            initial_version = await runtime.repo.get_project_version(project.current_version_id)
        await runtime.repo.remember_compatibility_run(
            user_id=user_id,
            idempotency_key=body.idempotency_key,
            project_id=project.id,
            session_id=session_id,
        )
        if planning_off or code_video_mode:
            workflow_id, activated_skill_ids = None, []
        else:
            workflow_id, activated_skill_ids = await _effective_skill_selection(
                runtime, project.id, requested_workflow, requested_activated,
            )
        language_values = _request_language_contract(
            body.objective,
            app_language=app_language,
            user_option=body.user_option,
            activated_skill_ids=activated_skill_ids,
        )
        run_context = {
            "user_option": body.user_option,
            "input_files": imported_input_files,
            "workflow_id": workflow_id,
            "activated_skill_ids": activated_skill_ids,
            "language_contract": language_values,
            "language_contract_version": 1,
            "automatic_video": False if planning_off or code_video_mode else body.automatic_video or workflow_id is not None,
            "planning_off": planning_off,
            "code_video_mode": code_video_mode,
        }
        await runtime.repo.remember_operation_result(
            project.id,
            _RUN_CONTEXT_OPERATION,
            _RUN_CONTEXT_KEY,
            json.dumps(run_context, ensure_ascii=False, sort_keys=True),
        )
        automatic_video = False if planning_off or code_video_mode else body.automatic_video or workflow_id is not None
        initial_prompt = (
            _code_video_prompt(
                objective=body.objective,
                project_id=project.id,
                base_project_version_id=initial_version.id,
                idempotency_key=body.idempotency_key,
                user_option=body.user_option,
                input_files=imported_input_files,
                language_contract=language_values,
            )
            if code_video_mode
            else _planning_off_prompt(
                objective=body.objective,
                project_id=project.id,
                base_project_version_id=initial_version.id,
                idempotency_key=body.idempotency_key,
                user_option=body.user_option,
                input_files=imported_input_files,
                language_contract=language_values,
            )
            if planning_off
            else _initial_video_build_prompt(
                objective=body.objective,
                project_id=project.id,
                base_project_version_id=initial_version.id,
                idempotency_key=body.idempotency_key,
                user_option=body.user_option,
                input_files=imported_input_files,
                workflow_id=workflow_id,
                activated_skill_ids=activated_skill_ids,
                language_contract=language_values,
            )
            if automatic_video
            else _initial_agent_prompt(
                objective=body.objective,
                project_id=project.id,
                base_project_version_id=initial_version.id,
                idempotency_key=body.idempotency_key,
                user_option=body.user_option,
                input_files=imported_input_files,
                workflow_id=workflow_id,
                activated_skill_ids=activated_skill_ids,
                language_contract=language_values,
            )
        )
        await _prompt_with_source_images(
            dsh,
            session_id,
            initial_prompt,
            imported_input_files,
        )
        return success(_run(project, session_id, [], run_context))
    except DeepSeekHarnessError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/runs")
async def list_runs(
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
) -> dict[str, Any]:
    rows = [
        _run_from_sidebar(item)
        for item in await runtime.repo.list_sidebar_runs(user_id, limit=_LIST_RUNS_LIMIT)
    ]
    return success(rows)


@router.get("/runs/{project_id}")
async def get_run(
    project_id: str,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
    dsh: Annotated[DeepSeekHarnessClient, Depends(get_deepseek_client)],
) -> dict[str, Any]:
    project, binding = await _project_and_binding(runtime, project_id, user_id)
    return success(await _snapshot(runtime, dsh, project, binding.session_id))


async def _continuation_context(runtime: VideoBuildRuntime, project, session_id: str) -> str:
    """Log durable execution references alongside a follow-up in its existing Session."""
    builds = await runtime.repo.list_builds(project.id)
    latest = max(builds, key=lambda item: item.created_at) if builds else None
    checkpoints = (
        await runtime.repo.list_build_checkpoints(project.id, latest.id) if latest else []
    )
    payload = {
        "thread_id": session_id,
        "project_id": project.id,
        "base_project_version_id": project.current_version_id,
        "latest_build": {"id": latest.id, "status": latest.status} if latest else None,
        "checkpoints": [
            {"id": item.id, "status": item.status, "phase": item.phase,
             "next_phase": item.next_phase, "artifact_version_ids": item.artifact_version_ids}
            for item in checkpoints
        ],
    }
    return (
        "\n\nDurable continuation context:\n"
        "Continue the same Harness Session using its conversation history, tool results, and "
        "compaction summary. Interpret short follow-ups against that history, not in isolation. "
        "For video work, inspect the bound project's current state and the relevant build or "
        "checkpoint before planning the next step. Read actual prior artifact contents and "
        "generation parameters when needed; reuse their exact version ids as dependencies. "
        "A checkpoint is execution state, not a replacement conversation or a new agent. "
        "Current Runtime state supersedes older progress notifications. Do not repeat completed "
        "work or create a replacement project. For ordinary chat, answer from Session history "
        "without starting video work.\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


@router.post("/runs/{project_id}/messages")
async def add_message(
    project_id: str,
    body: MessageBody,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
    dsh: Annotated[DeepSeekHarnessClient, Depends(get_deepseek_client)],
    app_language: Annotated[str | None, Header(alias="X-App-Language")] = None,
) -> dict[str, Any]:
    project, binding = await _project_and_binding(runtime, project_id, user_id)
    if body.thread_id and body.thread_id != binding.session_id:
        raise HTTPException(status_code=409, detail="thread_id does not match the project's bound Harness Session")
    if await runtime.repo.get_operation_result(
        project_id, "compat-message", body.idempotency_key,
    ) is not None:
        return success(await _run_with_context(
            runtime, project, binding.session_id, await _history(dsh, binding.session_id),
        ))
    try:
        imported_input_files = await _import_input_files(
            runtime, project_id, user_id, body.input_files,
        )
        if imported_input_files:
            project = await runtime.repo.get_project(project_id)
        # An empty draft may reuse a Session whose earlier turn was only an
        # inspection or acceptance test, and it has no Artifact or Build through
        # which the agent can recover project context. Re-apply the product-flow
        # contract only for automatic requests; conversation turns receive the
        # routing contract and remain free of unintended build side effects.
        is_empty_creation = (
            not [
                item for item in await runtime.repo.current_artifacts(project_id)
                if not item.type.startswith("source_")
            ]
            and not await runtime.repo.list_builds(project_id)
        )
        run_context = await _load_run_context(runtime, project_id)
        planning_off = _planning_off_requested(
            body.content,
            body.planning_off,
            current=bool(run_context.get("planning_off", False)),
        )
        code_video_mode = _code_video_mode_requested(
            body.code_video_mode,
            current=(
                bool(run_context.get("code_video_mode"))
                if "code_video_mode" in run_context else None
            ),
            planning_off=planning_off,
        )
        if planning_off and code_video_mode:
            raise HTTPException(
                status_code=422,
                detail="planning-off and Code-to-Video mode cannot be enabled together",
            )
        if planning_off or code_video_mode:
            requested_workflow, requested_activated = None, []
            workflow_id, activated_skill_ids = None, []
        else:
            requested_workflow, requested_activated = _requested_skill_selection(
                runtime, body.workflow_id, body.activated_skill_ids, body.content,
            )
            workflow_id, activated_skill_ids = await _effective_skill_selection(
                runtime, project_id, requested_workflow, requested_activated,
            )
        current_language = run_context.get("language_contract")
        current_language = current_language if isinstance(current_language, dict) else None
        language_values = _request_language_contract(
            body.content,
            app_language=app_language,
            user_option=body.user_option,
            activated_skill_ids=activated_skill_ids,
            current=current_language,
        )
        automatic_video = False if planning_off or code_video_mode else body.automatic_video or requested_workflow is not None
        prompt = (
            _code_video_prompt(
                objective=body.content,
                project_id=project_id,
                base_project_version_id=project.current_version_id,
                idempotency_key=body.idempotency_key,
                user_option=body.user_option,
                input_files=imported_input_files,
                language_contract=language_values,
            )
            if code_video_mode
            else _planning_off_prompt(
                objective=body.content,
                project_id=project_id,
                base_project_version_id=project.current_version_id,
                idempotency_key=body.idempotency_key,
                user_option=body.user_option,
                input_files=imported_input_files,
                language_contract=language_values,
            )
            if planning_off
            else _initial_video_build_prompt(
                objective=body.content,
                project_id=project_id,
                base_project_version_id=project.current_version_id,
                idempotency_key=body.idempotency_key,
                user_option=body.user_option,
                input_files=imported_input_files,
                workflow_id=workflow_id,
                activated_skill_ids=activated_skill_ids,
                language_contract=language_values,
            )
            if is_empty_creation and automatic_video
            else _initial_agent_prompt(
                objective=body.content,
                project_id=project_id,
                base_project_version_id=project.current_version_id,
                idempotency_key=body.idempotency_key,
                user_option=body.user_option,
                input_files=imported_input_files,
                workflow_id=workflow_id,
                activated_skill_ids=activated_skill_ids,
                language_contract=language_values,
            ) if is_empty_creation else body.content
        )
        if not is_empty_creation and not planning_off and not code_video_mode and (body.user_option or imported_input_files):
            prompt += (
                _UI_DEFAULTS_SEPARATOR
                + json.dumps(
                    {
                        "user_option": body.user_option or {},
                        "input_files": _prompt_input_files(imported_input_files),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        if not is_empty_creation and not planning_off and not code_video_mode:
            prompt = (
                f"{_AGENT_PROMPT_MARKER}\n"
                f"VISIBLE_USER_REQUEST_JSON: {json.dumps(body.content, ensure_ascii=False)}\n"
                + prompt
            )
            prompt += (
                "\nProject continuation policy: first decide whether the visible user message "
                "requests video project work. If it does not, answer directly without calling "
                "video_* tools. If it does, the request authorizes preview and apply without "
                "another confirmation. For an additional clip, use "
                "video_plan_patch_preview with api.provider.generate and chain media.concat. "
                "Do not stop after transcription: inspect its real segments, translate each "
                "segment with translated_texts ONLY if this request explicitly asks for translation. "
                "Otherwise use the original transcript language, not the UI language or a prior translation. Render using "
                "media.hyperframes_caption with the video, transcript, and authored caption_html. "
                "Never invent dialogue from the screenplay. "
                "Poll video_build_status until the requested output is complete; a queued "
                "build or a transcript alone does not satisfy a captioned-video request."
            )
            prompt += _selection_context(workflow_id, activated_skill_ids)
            prompt += "\n\n" + video_language_instruction(language_values)
        builds = await runtime.repo.list_builds(project_id)
        latest = max(builds, key=lambda item: item.created_at) if builds else None
        stopped = latest is not None and latest.status == "cancelled"
        if stopped and not body.resume_builds:
            raise HTTPException(status_code=409, detail={
                "code": "task_stopped_resume_required",
                "message": "Task stopped. Confirm resume before sending a continuation.",
            })
        if stopped:
            # Only the latest stopped build is eligible, never historical cancelled plans.
            if latest is not None:
                try:
                    await runtime.resume_cancelled_build(project_id=project_id, build_id=latest.id)
                except ProjectVersionConflict as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                checkpoints = await runtime.repo.list_build_checkpoints(project_id, latest.id)
                active = [item.id for item in checkpoints if item.status in {"pending", "planning"}]
                prompt += (
                    f"\nUser confirmed resume of existing build {latest.id}, project {project_id}. "
                    "Do not create a replacement build or repeat completed tasks. Inspect existing "
                    "artifacts and resume the pending checkpoint using the user's message. "
                    f"Active checkpoints: {json.dumps(active)}."
                )
        # This context enters the durable Harness log with the user message;
        # model replay and compaction therefore see the same project references.
        prompt += await _continuation_context(runtime, project, binding.session_id)
        # PlanPatch tools run during the Harness turn. Persist the mode before
        # prompting so Runtime's execution gate observes the user's current toggle.
        if bool(run_context.get("code_video_mode", False)) != code_video_mode:
            run_context["code_video_mode"] = code_video_mode
            await runtime.repo.replace_operation_result(
                project_id,
                _RUN_CONTEXT_OPERATION,
                _RUN_CONTEXT_KEY,
                json.dumps(run_context, ensure_ascii=False, sort_keys=True),
            )
        await _prompt_with_source_images(
            dsh, binding.session_id, prompt, imported_input_files,
        )
        run_context.update({
            "workflow_id": workflow_id,
            "activated_skill_ids": activated_skill_ids,
            "language_contract": language_values,
            "language_contract_version": 1,
            "automatic_video": automatic_video,
            "planning_off": planning_off,
            "code_video_mode": code_video_mode,
        })
        if body.user_option is not None:
            run_context["user_option"] = body.user_option
        if imported_input_files:
            run_context["input_files"] = [
                *list(run_context.get("input_files") or []),
                *imported_input_files,
            ]
        await runtime.repo.replace_operation_result(
            project_id,
            _RUN_CONTEXT_OPERATION,
            _RUN_CONTEXT_KEY,
            json.dumps(run_context, ensure_ascii=False, sort_keys=True),
        )
        await runtime.repo.remember_operation_result(
            project_id, "compat-message", body.idempotency_key, binding.session_id,
        )
    except DeepSeekHarnessError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return success(await _run_with_context(
        runtime, project, binding.session_id, await _history(dsh, binding.session_id),
    ))


@router.post("/runs/{project_id}/resume")
async def resume_run(
    project_id: str,
    body: MessageBody,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
    dsh: Annotated[DeepSeekHarnessClient, Depends(get_deepseek_client)],
    app_language: Annotated[str | None, Header(alias="X-App-Language")] = None,
) -> dict[str, Any]:
    return await add_message(
        project_id, body.model_copy(update={"resume_builds": True}),
        user_id, runtime, dsh, app_language,
    )


@router.get("/runs/{project_id}/messages")
async def list_messages(
    project_id: str,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
    dsh: Annotated[DeepSeekHarnessClient, Depends(get_deepseek_client)],
) -> dict[str, Any]:
    project, binding = await _project_and_binding(runtime, project_id, user_id)
    return success(_messages(project.id, await _history(dsh, binding.session_id)))


def _compat_event(project_id: str, event: dict[str, Any]) -> dict[str, Any] | None:
    event_type = event.get("type")
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    if event_type in {"user/message", "assistant/message"}:
        role = "user" if event_type == "user/message" else "assistant"
        source = data.get("source")
        if role == "user" and isinstance(source, dict) and source.get("kind") != "user":
            return None
        content_source = data.get("content") if role == "user" else (
            data.get("message", {}).get("content")
            if isinstance(data.get("message"), dict) else ""
        )
        mapped_type = "chat.message.created"
        milliseconds = event.get("time") if isinstance(event.get("time"), (int, float)) else 0
        created_at = datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()
        visible_content = (
            _visible_user_text(_text(content_source)) if role == "user"
            else _text(content_source)
        )
        if not visible_content:
            return None
        payload = {"message": {
            "id": f"dsh-{project_id}-{event.get('seq', 0)}",
            "run_id": project_id,
            "role": role,
            "content": visible_content,
            "sequence": event.get("seq", 0) + 1,
            "metadata": {"source": "deepseek-harness"},
            "created_at": created_at,
        }}
    elif event_type == "assistant/chunk":
        chunk = data.get("chunk") if isinstance(data.get("chunk"), dict) else {}
        usage = _usage_payload(data)
        if usage is not None:
            mapped_type, payload = "llm.usage", usage
        # Only final-answer text is streamed into chat. Reasoning blocks remain
        # private; the production-progress UI exposes phase/tool/build telemetry.
        elif chunk.get("type") != "text-delta" or not isinstance(chunk.get("text"), str):
            return None
        else:
            mapped_type = "agent.message.delta"
            payload = {
                "delta": chunk["text"],
                "message_id": f"stream-{project_id}-turn-{data.get('turn', 0)}",
            }
    elif event_type == "tool/call":
        mapped_type = "agent.tool.started"
        payload = {
            "turn": data.get("turn"),
            "step": data.get("step"),
            "call_id": str(data.get("callId") or ""),
            "tool": str(data.get("name") or "unknown"),
            "input": _json_arguments(data.get("arguments")),
        }
    elif event_type == "tool/result":
        call_id, output, is_error = _tool_result(data)
        mapped_type = "agent.tool.completed"
        payload = {
            "turn": data.get("turn"),
            "step": data.get("step"),
            "call_id": call_id,
            "output": output,
            "is_error": is_error,
            "error": output if is_error else None,
        }
    elif event_type == "turn/start":
        mapped_type, payload = "agent.started", data
    elif event_type == "turn/end":
        status = _status([event])
        mapped_type = {
            "completed": "run.completed",
            "failed": "run.failed",
            "cancelled": "run.cancelled",
        }.get(status, "run.completed")
        payload = {**data, "status": status}
        if status == "failed":
            # The studio notice consumes a top-level string, not Harness's
            # nested turn/end reason. Keep the provider's actionable message.
            reason = data.get("reason") or {}
            error = reason.get("error")
            if isinstance(error, dict):
                payload["error"] = error.get("message") or "Model request failed"
                payload["error_code"] = error.get("code")
            elif isinstance(error, str) and error:
                payload["error"] = error
            else:
                payload["error"] = "Model request failed"
    elif event_type == "request/header":
        header = data.get("header") if isinstance(data.get("header"), dict) else {}
        config = header.get("config") if isinstance(header.get("config"), dict) else {}
        mapped_type = "llm.request.started"
        payload = {
            "provider": config.get("provider"),
            "model": config.get("model"),
            "reasoning_effort": config.get("reasoningEffort"),
            "tool_count": len(header.get("tools", [])) if isinstance(header.get("tools"), list) else 0,
        }
    elif event_type == "request/context":
        mapped_type = "llm.context.ready"
        payload = {
            "provider": data.get("provider"),
            "model": data.get("model"),
            "context_window": data.get("contextWindow"),
        }
    elif event_type == "step/start":
        mapped_type, payload = "agent.step.started", data
    elif event_type == "step/end":
        mapped_type, payload = "agent.step.completed", data
    elif event_type == "agent/inbox/spliced":
        inserted = data.get("inserted") if isinstance(data.get("inserted"), list) else []
        mapped_type = "agent.context.updated"
        payload = {
            "target": data.get("target"),
            "inserted_count": len(inserted),
            "removed_count": int(data.get("removedCount") or 0),
        }
    else:
        mapped_type, payload = f"deepseek.{event_type}", data
    milliseconds = event.get("time") if isinstance(event.get("time"), (int, float)) else 0
    sequence = event.get("seq", 0) + 1
    return {
        "id": f"dsh-{project_id}-{event.get('seq', 0)}",
        "run_id": project_id,
        "sequence": sequence,
        "type": mapped_type,
        "payload": payload,
        "created_at": datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat(),
    }


@router.get("/runs/{project_id}/event-log")
async def event_log(
    project_id: str,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
    dsh: Annotated[DeepSeekHarnessClient, Depends(get_deepseek_client)],
) -> dict[str, Any]:
    _project, binding = await _project_and_binding(runtime, project_id, user_id)
    return success([
        mapped for item in await _history(dsh, binding.session_id)
        if (mapped := _compat_event(project_id, item)) is not None
    ])


@router.get("/runs/{project_id}/events", response_model=None)
async def stream_events(
    request: Request,
    project_id: str,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
    dsh: Annotated[DeepSeekHarnessClient, Depends(get_deepseek_client)],
    after: int = Query(default=0, ge=0),
) -> StreamingResponse:
    _project, binding = await _project_and_binding(runtime, project_id, user_id)

    async def stream() -> AsyncIterator[str]:
        cursor = after
        while not await request.is_disconnected():
            events = [
                item for item in await _history(dsh, binding.session_id)
                if isinstance(item.get("seq"), int) and item["seq"] + 1 > cursor
            ]
            if not events:
                yield ": heartbeat\n\n"
                await asyncio.sleep(1)
                continue
            for item in events:
                mapped = _compat_event(project_id, item)
                if mapped is None:
                    cursor = item["seq"] + 1
                    continue
                cursor = mapped["sequence"]
                yield (
                    f"id: {mapped['sequence']}\nevent: {mapped['type']}\n"
                    f"data: {json.dumps(mapped, ensure_ascii=False)}\n\n"
                )

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/runs/{project_id}/cancel")
async def cancel_run(
    project_id: str,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
    dsh: Annotated[DeepSeekHarnessClient, Depends(get_deepseek_client)],
) -> dict[str, Any]:
    project, binding = await _project_and_binding(runtime, project_id, user_id)
    # Stop durable execution before cancelling the conversational loop, including
    # when the DeepSeek service is temporarily unreachable.
    async def stop_builds() -> None:
        for build in await runtime.repo.list_builds(project_id):
            if build.status in {"queued", "running", "waiting_external", "waiting_agent"}:
                await runtime.cancel_build(project_id=project_id, build_id=build.id)
    lock = runtime.session_control_locks.setdefault(project_id, asyncio.Lock())
    async with lock:
        await stop_builds()
        try:
            await dsh.cancel(binding.session_id)
        except DeepSeekHarnessError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            # Catch a tool submission that completed during session cancellation.
            await stop_builds()
    return success(await _run_with_context(
        runtime, project, binding.session_id, await _history(dsh, binding.session_id),
    ))


@router.post("/runs/{project_id}/artifacts/{version_id}/select")
async def select_artifact(
    project_id: str,
    version_id: str,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
) -> dict[str, Any]:
    project, _binding = await _project_and_binding(runtime, project_id, user_id)
    current = await runtime.repo.get_project_version(project.current_version_id)
    already_selected = next(
        (item for item in await runtime.repo.current_artifacts(project_id) if item.id == version_id),
        None,
    )
    if already_selected is not None:
        artifact, version = already_selected, current
    else:
        try:
            artifact, version = await runtime.select_artifact(
                project_id=project_id,
                version_id=version_id,
                base_project_version_id=current.id,
                idempotency_key=f"compat-select:{current.id}:{version_id}",
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="artifact version not found") from exc
    return success({
        "project_id": project_id,
        "type": artifact.type,
        "artifact_version_id": artifact.id,
        "updated_at": version.created_at.isoformat(),
    })


@router.get("/runs/{project_id}/token-usage")
async def token_usage(
    project_id: str,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
    dsh: Annotated[DeepSeekHarnessClient, Depends(get_deepseek_client)],
) -> dict[str, Any]:
    _project, binding = await _project_and_binding(runtime, project_id, user_id)
    records = [
        usage for event in await _history(dsh, binding.session_id)
        if (usage := _usage_payload(
            event.get("data") if isinstance(event.get("data"), dict) else {}
        )) is not None
    ]
    input_tokens = sum(int(item["input_tokens"]) for item in records)
    output_tokens = sum(int(item["output_tokens"]) for item in records)
    cached_tokens = sum(int(item["cached_tokens"]) for item in records)
    reasoning_tokens = sum(int(item["reasoning_tokens"]) for item in records)
    total_tokens = sum(int(item["total_tokens"]) for item in records)
    return success({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "requested_tokens": 0,
        "calls": len(records),
        "failed_calls": 0,
        "retry_attempts": 0,
        "by_scope": {"deepseek-harness": {
            "calls": len(records),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }} if records else {},
        "by_task": [],
        "records": records,
        "attempt_records": [],
    })


@router.get("/skills")
async def list_skills(
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
) -> dict[str, Any]:
    views = runtime.skills.prompt_view()
    for view in views:
        if view.get("execution_kind") == "agent_plan_patch" and not (
            runtime.staged_planning_enabled and runtime.continuous_plan_patch_enabled
        ):
            view["available"] = False
            view["unavailable_reason"] = "Workflow requires continuous PlanPatch planning to be enabled"
    return success(views)


_UPLOAD_HTTP_STATUS = {
    BusinessExceptionCode.FILE_TOO_LARGE: 413,
    BusinessExceptionCode.UNSUPPORTED_FILE_FORMAT: 415,
    BusinessExceptionCode.FILE_UPLOAD_FAILED: 400,
}


@router.post("/uploads")
async def upload_files(
    user_id: Annotated[str, Depends(identity)],
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    try:
        images, audio_files, video_files = await process_uploaded_files(files)
    except BusinessException as exc:
        payload = exc.to_dict()
        payload["message"] = exc.detail
        return JSONResponse(
            status_code=_UPLOAD_HTTP_STATUS.get(exc.error_code, 400),
            content=payload,
        )
    uploaded = []
    for kind, items in (("image", images), ("audio", audio_files), ("video", video_files)):
        for item in items:
            metadata = item.model_dump(mode="json")
            metadata["upload_receipt"] = _sign_uploaded_file(user_id, kind, item.url)
            uploaded.append({
                "type": kind,
                "url": item.url,
                "filename": getattr(item, "filename", None),
                "metadata": metadata,
            })
    return success({"files": uploaded})


async def _studio_project(
    runtime: VideoBuildRuntime,
    session_id: str,
    user_id: str,
):
    try:
        return await runtime.repo.project_for_session(session_id, user_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Studio project not found") from exc


@studio_router.post("/projects")
async def studio_create_project(
    body: StudioProjectBody,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
    dsh: Annotated[DeepSeekHarnessClient, Depends(get_deepseek_client)],
    app_language: Annotated[str | None, Header(alias="X-App-Language")] = None,
) -> dict[str, Any]:
    created = await create_run(
        CreateRunBody(
            objective=body.objective,
            idempotency_key=f"studio-create:{uuid4()}",
            thread_id=body.thread_id,
            user_option=body.user_option,
            input_files=body.input_files,
            workflow_id=body.workflow_id,
            activated_skill_ids=body.activated_skill_ids,
        ),
        user_id,
        runtime,
        dsh,
        app_language,
    )
    return success({"project": created["data"], "suggested_plan": []})


@studio_router.get("/projects/{session_id}")
async def studio_get_project(
    session_id: str,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
    dsh: Annotated[DeepSeekHarnessClient, Depends(get_deepseek_client)],
) -> dict[str, Any]:
    project, binding = await _studio_project(runtime, session_id, user_id)
    return success(await _snapshot(runtime, dsh, project, binding.session_id))


@studio_router.post("/projects/{session_id}/commands")
async def studio_add_command(
    session_id: str,
    body: StudioProjectBody,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
    dsh: Annotated[DeepSeekHarnessClient, Depends(get_deepseek_client)],
    app_language: Annotated[str | None, Header(alias="X-App-Language")] = None,
) -> dict[str, Any]:
    project, binding = await _studio_project(runtime, session_id, user_id)
    imported_input_files = await _import_input_files(
        runtime, project.id, user_id, body.input_files,
    )
    if imported_input_files:
        project = await runtime.repo.get_project(project.id)
    requested_workflow, requested_activated = _requested_skill_selection(
        runtime, body.workflow_id, body.activated_skill_ids, body.objective,
    )
    workflow_id, activated_skill_ids = await _effective_skill_selection(
        runtime, project.id, requested_workflow, requested_activated,
    )
    run_context = await _load_run_context(runtime, project.id)
    current_language = run_context.get("language_contract")
    current_language = current_language if isinstance(current_language, dict) else None
    language_values = _request_language_contract(
        body.objective,
        app_language=app_language,
        user_option=body.user_option,
        activated_skill_ids=activated_skill_ids,
        current=current_language,
    )
    try:
        prompt = body.objective
        if body.user_option or imported_input_files:
            prompt += (
                _UI_DEFAULTS_SEPARATOR
                + json.dumps(
                    {
                        "user_option": body.user_option or {},
                        "input_files": _prompt_input_files(imported_input_files),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        await _prompt_with_source_images(
            dsh,
            binding.session_id,
            prompt + _selection_context(workflow_id, activated_skill_ids)
            + "\n\n" + video_language_instruction(language_values),
            imported_input_files,
        )
        run_context.update({
            "workflow_id": workflow_id,
            "activated_skill_ids": activated_skill_ids,
            "language_contract": language_values,
            "language_contract_version": 1,
        })
        await runtime.repo.remember_operation_result(
            project.id, _RUN_CONTEXT_OPERATION, _RUN_CONTEXT_KEY,
            json.dumps(run_context, ensure_ascii=False, sort_keys=True),
        )
    except DeepSeekHarnessError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return success({
        "project": await _run_with_context(
            runtime, project, binding.session_id, await _history(dsh, binding.session_id),
        ),
        "suggested_plan": [],
    })


@studio_router.get("/projects/{session_id}/events")
async def studio_events(
    session_id: str,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
    dsh: Annotated[DeepSeekHarnessClient, Depends(get_deepseek_client)],
) -> dict[str, Any]:
    project, binding = await _studio_project(runtime, session_id, user_id)
    return success([
        mapped for item in await _history(dsh, binding.session_id)
        if (mapped := _compat_event(project.id, item)) is not None
    ])


@studio_router.get("/projects/{session_id}/skills")
async def studio_project_skills(
    session_id: str,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
) -> dict[str, Any]:
    project, _binding = await _studio_project(runtime, session_id, user_id)
    return success([
        item.model_dump(mode="json")
        for item in await runtime.list_project_skill_locks(project.id)
    ])


@studio_router.post("/projects/{session_id}/skills/{skill_id}/enable")
async def studio_set_project_skill(
    session_id: str,
    skill_id: str,
    body: StudioSkillEnableBody,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
) -> dict[str, Any]:
    project, _binding = await _studio_project(runtime, session_id, user_id)
    try:
        lock = await runtime.set_project_skill_enabled(
            project_id=project.id,
            skill_id=skill_id,
            enabled=body.enabled,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return success(lock.model_dump(mode="json"))


@studio_router.post("/projects/{session_id}/artifacts/{version_id}/select")
async def studio_select_artifact(
    session_id: str,
    version_id: str,
    user_id: Annotated[str, Depends(identity)],
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
) -> dict[str, Any]:
    project, _binding = await _studio_project(runtime, session_id, user_id)
    return await select_artifact(project.id, version_id, user_id, runtime)


@studio_router.get("/skills")
async def studio_list_skills(
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
) -> dict[str, Any]:
    return await list_skills(runtime)


@studio_router.post("/skills/install")
async def studio_install_skill(
    runtime: Annotated[VideoBuildRuntime, Depends(get_runtime)],
    bundle: Annotated[UploadFile, File()],
    overwrite: Annotated[bool, Form()] = False,
) -> dict[str, Any]:
    from app.chat.v2.skill_install import (
        MAX_BUNDLE_BYTES,
        SkillInstallError,
        extract_skill_archive,
        install_skill_directory,
    )

    data = await bundle.read(MAX_BUNDLE_BYTES + 1)
    try:
        with tempfile.TemporaryDirectory(prefix="cuti-skill-upload-") as directory:
            source = extract_skill_archive(data, Path(directory))
            name = install_skill_directory(
                source,
                runtime.skills.external_root,
                catalog=runtime.skills.catalog,
                registry=runtime.skills.capabilities,
                overwrite=overwrite,
            )
        skill_count = await runtime.reload_skills()
    except (SkillInstallError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return success({"name": name, "skill_count": skill_count})
