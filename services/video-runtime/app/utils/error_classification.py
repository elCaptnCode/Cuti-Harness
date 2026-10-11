"""Unified generation-failure reason classifier.

Maps raw errors from the underlying layer (provider API / LLM / tool) into a limited set of "official categories",
exposing only user-understandable reason text that does not leak internal info (provider name, model name, stack, status-code details).

Usage:
    cat = classify_failure(raw_error_msg)
    reason = await user_facing_reason_async(raw_error_msg, lang="zh")

raw_error_msg is still only for internal logs/debug persistence and is never returned to the frontend directly.
"""
from enum import Enum
from typing import Optional

from app.utils.i18n import get_i18n_message, get_i18n_message_async


class FailureCategory(str, Enum):
    """Official failure categories exposed externally (deliberately coarse-grained to avoid leaking internal details)."""

    RATE_LIMITED = "rate_limited"            # high current request volume / rate limit triggered
    CONTENT_MODERATION = "content_moderation"  # content moderation failed / safety policy triggered
    QUOTA_EXCEEDED = "quota_exceeded"        # insufficient balance / quota
    SERVICE_UNAVAILABLE = "service_unavailable"  # service temporarily unavailable / timeout
    INVALID_INPUT = "invalid_input"          # input / prompt not supported
    UNKNOWN = "unknown"                      # other unknown reasons
    QUALITY_REJECTED = "quality_rejected"    # a Video Runtime validator rejected an artifact; never returned by classify_failure


# keyword table (all matched lowercase). Order in classify_failure decides priority.
_MODERATION_KEYWORDS = [
    "moderation", "content policy", "content_policy", "safety", "safety system",
    "sensitive", "violat", "policy violation", "nsfw", "prohibited", "not allowed",
    "blocked by", "flagged", "审核", "敏感", "违规", "违禁", "不合规", "拦截",
]
# provider/platform-side account balance (not user credits) -> uniformly presented as service-busy, to avoid users thinking their own credits are exhausted
_PROVIDER_BALANCE_KEYWORDS = [
    "账户余额", "请充值", "account balance", "insufficient balance",
    "insufficient credit", "out of credit", "no credit", "billing account", "欠费",
    "top up your account", "please top up",
]
_QUOTA_KEYWORDS = [
    "insufficient quota", "quota exceeded", "exceeded your quota",
    "402", "额度", "配额", "积分不足",
]
_RATE_LIMIT_KEYWORDS = [
    "rate limit", "rate_limit", "too many requests", "429", "throttl",
    "限流", "请求过于频繁", "频繁", "使用量",
]
_SERVICE_KEYWORDS = [
    "500", "502", "503", "504", "deadline_exceeded", "timeout", "timed out",
    "capacity", "temporarily unavailable", "service unavailable", "overloaded",
    "connection", "unavailable", "gateway", "暂时不可用", "超时", "繁忙",
]
_INVALID_INPUT_KEYWORDS = [
    "invalid", "bad request", "400", "422", "unsupported", "not supported",
    "unprocessable", "validation", "参数", "不支持", "格式错误",
]
# media input cannot be fetched remotely (local/private-network URL, download failure) -> an input/config problem; must never be misclassified as content moderation by "not allowed".
# typical: WaveSpeed "Invalid image URL: private or local network URLs are not allowed" (local storage without media egress).
_MEDIA_URL_KEYWORDS = [
    "private or local network", "local network url", "invalid image url",
    "invalid video url", "invalid audio url", "failed to download",
    "could not download", "unable to fetch", "url is not accessible",
    "无法访问", "无法下载",
]


def classify_failure(raw_error: Optional[str]) -> FailureCategory:
    """Classify by the raw error message.

    Priority: media-URL-unreachable > content moderation > quota > rate limit > service unavailable > invalid input > unknown.
    (Media-URL-unreachable is highest, to keep its "not allowed" from being swallowed by content moderation; moderation/quota next, to avoid being misclassified as rate-limit by generic words like "unavailable".)
    """
    if not raw_error:
        return FailureCategory.UNKNOWN
    text = raw_error.lower()

    def _hit(keywords: list) -> bool:
        return any(kw in text for kw in keywords)

    if _hit(_MEDIA_URL_KEYWORDS):
        return FailureCategory.INVALID_INPUT
    if _hit(_MODERATION_KEYWORDS):
        return FailureCategory.CONTENT_MODERATION
    if _hit(_PROVIDER_BALANCE_KEYWORDS):
        return FailureCategory.SERVICE_UNAVAILABLE
    if _hit(_QUOTA_KEYWORDS):
        return FailureCategory.QUOTA_EXCEEDED
    if _hit(_RATE_LIMIT_KEYWORDS):
        return FailureCategory.RATE_LIMITED
    if _hit(_SERVICE_KEYWORDS):
        return FailureCategory.SERVICE_UNAVAILABLE
    if _hit(_INVALID_INPUT_KEYWORDS):
        return FailureCategory.INVALID_INPUT
    return FailureCategory.UNKNOWN


def category_i18n_key(category: FailureCategory) -> str:
    """The user-facing i18n key for this category."""
    return f"generation_failed.reason.{category.value}"


def user_facing_reason(
    raw_error: Optional[str] = None,
    *,
    category: Optional[FailureCategory] = None,
    lang: Optional[str] = None,
) -> str:
    """Synchronously return the user-visible official reason text (no internal info).

    Uses category directly if passed; otherwise classifies from raw_error.
    """
    cat = category or classify_failure(raw_error)
    return get_i18n_message(category_i18n_key(cat), default=_default_reason(cat), lang=lang)


async def user_facing_reason_async(
    raw_error: Optional[str] = None,
    *,
    category: Optional[FailureCategory] = None,
    lang: Optional[str] = None,
) -> str:
    """Async version, see user_facing_reason."""
    cat = category or classify_failure(raw_error)
    return await get_i18n_message_async(
        category_i18n_key(cat), default=_default_reason(cat), lang=lang
    )


# blacklist that, once matched, marks text as "possibly leaking internal info" (all matched lowercase).
# used as a safety net for tool-LLM-generated error_msg: if any item matches, fall back to the official category text and never pass provider/model/technical details to the user.
_LEAK_KEYWORDS = [
    # provider / platform / model names
    "pollo", "wavespeed", "seedance", "seedream", "nano banana", "nano_banana",
    "kling", "sora", "openai", "gpt", "gemini", "flux", "wan2", "ltx", "suno",
    "minimax", "midjourney", "runway", "luma", "jimeng", "即梦", "可灵",
    # technical details / protocol / code
    "http", "api", "sdk", "traceback", "exception", "status code", "json",
    "endpoint", "request id", "url", "uuid", "504", "503", "502", "500",
    "429", "401", "403", "404", "422", "400",
    # Chinese technical terms
    "异常", "堆栈", "接口", "状态码", "错误码", "调用失败", "报错",
    # provider account balance (not user credits)
    "余额", "充值", "账户", "欠费", "额度不足",
    "insufficient credit", "top up your account", "please top up",
]


def sanitize_user_facing_reason(
    text: Optional[str],
    *,
    fallback: str,
    max_len: int = 80,
) -> str:
    """Post-filter tool-LLM-generated error_msg to prevent leaks (llm + guard strategy).

    - if text is empty / matches the leak blacklist / is too long (likely carrying the raw error) -> return fallback (official category text);
    - otherwise return the text with the leading status symbol removed.
    fallback is usually the category text from user_facing_reason(category=...).
    """
    if not text or not text.strip():
        return fallback
    cleaned = text.strip().lstrip("❌✅⚠️ ").strip()
    if not cleaned:
        return fallback
    low = cleaned.lower()
    if any(kw in low for kw in _LEAK_KEYWORDS):
        return fallback
    if len(cleaned) > max_len:
        return fallback
    return cleaned


def _default_reason(category: FailureCategory) -> str:
    """i18n fallback text (Chinese), used only when the locale is missing."""
    return {
        FailureCategory.RATE_LIMITED: "当前生成需求较多，请稍后重试。",
        FailureCategory.CONTENT_MODERATION: "内容未通过安全审核，请调整描述后重试。",
        FailureCategory.QUOTA_EXCEEDED: "当前生成资源紧张，请稍后重试。",
        FailureCategory.SERVICE_UNAVAILABLE: "生成服务暂时繁忙，请稍后重试。",
        FailureCategory.INVALID_INPUT: "当前内容暂不被支持，请调整描述后重试。",
        FailureCategory.UNKNOWN: "生成失败，请稍后重试。",
        FailureCategory.QUALITY_REJECTED: "生成结果未通过质量校验，请调整描述后重试。",
    }.get(category, "生成失败，请稍后重试。")
