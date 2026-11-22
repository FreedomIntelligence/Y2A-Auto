from __future__ import annotations

import json
from typing import Any

from .models import (
    EVENT_LOGIN_LOCKED,
    EVENT_LOGIN_SUCCESS,
    EVENT_MONITOR_ALERT,
    EVENT_MONITOR_RUN_FINISHED,
    EVENT_QR_LOGIN_FAILED,
    EVENT_QR_LOGIN_SUCCESS,
    EVENT_TASK_ADDED,
    EVENT_TASK_COMPLETED,
    EVENT_TASK_FAILED,
    NotificationEvent,
    NotificationMessage,
)


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _truncate(text: str, limit: int = 240) -> str:
    clean = _as_text(text)
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 1)] + "…"


def _upload_target_label(upload_target: Any) -> str:
    target = _as_text(upload_target).lower()
    if target == "both":
        return "AcFun + bilibili"
    if target == "bilibili":
        return "bilibili"
    return "AcFun"


def _task_title(payload: dict[str, Any]) -> str:
    for key in ("video_title_translated", "video_title_original", "title"):
        value = _as_text(payload.get(key))
        if value:
            return value
    return "未命名任务"


def _task_platform_result(payload: dict[str, Any]) -> str:
    succeeded = []
    if payload.get("acfun_uploaded"):
        succeeded.append("AcFun")
    if payload.get("bilibili_uploaded"):
        succeeded.append("bilibili")
    if succeeded:
        return "、".join(succeeded)
    return _upload_target_label(payload.get("upload_target"))


def _pretty_error_text(value: Any) -> str:
    text = _truncate(_as_text(value), 300)
    return text or "未提供错误详情"


def _markdown_lines(*lines: str) -> str:
    return "\n".join(line for line in lines if _as_text(line))


def _kv(label: str, value: Any) -> str:
    """格式化一个键值行：**标签** 值"""
    text = _as_text(value)
    if not text:
        return ""
    return f"**{label}：**{text}"


def _section_block(*kv_lines: str) -> str:
    """将多行键值组装成引用块。"""
    return "\n".join(f"> {line}" for line in kv_lines if _as_text(line))


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _monitor_video_line(index: int, video: Any) -> str:
    if not isinstance(video, dict):
        return ""
    title = _truncate(_as_text(video.get("title")), 80) or _as_text(video.get("id"))
    score = _as_int(video.get("ai_score"))
    prefix = f"[{score}分] " if score is not None else ""
    reason = _truncate(_as_text(video.get("ai_reason")), 40)
    suffix = f"（{reason}）" if reason else ""
    return f"{index}. {prefix}{title}{suffix}"


def _build_monitor_message(event_type: str, payload: dict[str, Any]) -> NotificationMessage:
    """监控运行摘要 / 监控异常。payload 由 youtube_monitor.run_monitor 生成。"""
    name = _as_text(payload.get("config_name")) or f"监控 {_as_text(payload.get('config_id'))}"
    added = _as_int(payload.get("added_count")) or 0
    is_alert = event_type == EVENT_MONITOR_ALERT
    alert_reason = _truncate(_as_text(payload.get("alert_reason")), 300)

    ai_line = ""
    if payload.get("ai_enabled"):
        if payload.get("ai_error"):
            ai_line = _kv("AI 选题", f"失败：{_truncate(_as_text(payload.get('ai_error')), 200)}")
        else:
            ai_line = _kv(
                "AI 选题",
                f"新候选 {_as_int(payload.get('ai_candidates')) or 0} 个，"
                f"保留 {_as_int(payload.get('ai_kept')) or 0} 个（门槛 {_as_int(payload.get('ai_min_score'))} 分）",
            )

    body = _section_block(
        _kv("监控", name),
        _kv("异常", alert_reason) if is_alert else "",
        _kv("搜索结果", f"{_as_int(payload.get('fetched_count')) or 0} 个"),
        _kv("规则过滤后", f"{_as_int(payload.get('filtered_count')) or 0} 个"),
        ai_line,
        _kv("加入任务队列", f"{added} 个"),
        _kv("时间", _as_text(payload.get("occurred_at"))),
    )

    lines = []
    added_videos = payload.get("added_videos") if isinstance(payload.get("added_videos"), list) else []
    for index, video in enumerate(added_videos[:10], 1):
        line = _monitor_video_line(index, video)
        if line:
            lines.append(f"> {line}")
    rejected = payload.get("ai_rejected_samples") if isinstance(payload.get("ai_rejected_samples"), list) else []
    rejected_lines = []
    for index, video in enumerate(rejected[:5], 1):
        line = _monitor_video_line(index, video)
        if line:
            rejected_lines.append(f"> {line}")

    if is_alert:
        title = "Y2A-Auto ⚠️ 监控异常"
        summary = f"{name} | {alert_reason or '请检查监控配置'}"
        heading = "**⚠️ 监控异常**"
    else:
        title = "Y2A-Auto 🔎 监控运行完成"
        summary = f"{name} | 新增任务 {added} 个"
        heading = "**🔎 监控运行完成**"

    parts = [heading, "", body]
    if lines:
        parts += ["", "> **本轮加入的视频：**", *lines]
    if rejected_lines:
        parts += ["", "> **AI 淘汰的部分候选：**", *rejected_lines]
    return NotificationMessage(title=title, summary=summary, markdown=_markdown_lines(*parts))


def build_notification_message(event: NotificationEvent) -> NotificationMessage:
    payload = event.as_payload()
    event_type = event.event_type

    if event_type == EVENT_TASK_ADDED:
        title = "Y2A-Auto 📋 任务已添加"
        summary = f"{payload.get('task_id', '')[:8]} | {_upload_target_label(payload.get('upload_target'))}"
        body = _section_block(
            _kv("任务 ID", f"`{_as_text(payload.get('task_id'))}`"),
            _kv("投稿目标", _upload_target_label(payload.get("upload_target"))),
            _kv("YouTube URL", _truncate(_as_text(payload.get("youtube_url")), 500)),
            _kv("时间", _as_text(payload.get("occurred_at"))),
        )
        markdown = _markdown_lines("**📋 任务已添加**", "", body)
        return NotificationMessage(title=title, summary=summary, markdown=markdown)

    if event_type == EVENT_TASK_COMPLETED:
        task_title = _task_title(payload)
        platform_result = _task_platform_result(payload)
        title = "Y2A-Auto ✅ 任务已完成"
        summary = f"{task_title} | {platform_result}"
        body = _section_block(
            _kv("视频标题", _truncate(task_title, 120)),
            _kv("任务 ID", f"`{_as_text(payload.get('task_id'))}`"),
            _kv("投稿结果", platform_result),
            _kv("投稿目标", _upload_target_label(payload.get("upload_target"))),
            _kv("时间", _as_text(payload.get("occurred_at"))),
        )
        markdown = _markdown_lines("**✅ 任务已完成**", "", body)
        return NotificationMessage(title=title, summary=summary, markdown=markdown)

    if event_type == EVENT_TASK_FAILED:
        task_title = _task_title(payload)
        error_text = _pretty_error_text(payload.get("error_message"))
        title = "Y2A-Auto ❌ 任务失败"
        summary = f"{task_title} | {error_text}"
        body = _section_block(
            _kv("视频标题", _truncate(task_title, 120)),
            _kv("任务 ID", f"`{_as_text(payload.get('task_id'))}`"),
            _kv("当前状态", _as_text(payload.get("status")) or "failed"),
            _kv("投稿目标", _upload_target_label(payload.get("upload_target"))),
        )
        markdown = _markdown_lines("**❌ 任务失败**", "", body, "", f"> **错误详情：**", f"> {error_text}", "", f"> {_kv('时间', _as_text(payload.get('occurred_at')))}")
        return NotificationMessage(title=title, summary=summary, markdown=markdown)

    if event_type == EVENT_LOGIN_SUCCESS:
        ip = _as_text(payload.get("ip_address")) or "unknown"
        occurred_at = _as_text(payload.get("occurred_at"))
        title = "Y2A-Auto 🔐 后台登录成功"
        summary = f"{ip} | {occurred_at}"
        body = _section_block(
            _kv("来源 IP", ip),
            _kv("时间", occurred_at),
        )
        markdown = _markdown_lines("**🔐 后台登录成功**", "", body)
        return NotificationMessage(title=title, summary=summary, markdown=markdown)

    if event_type == EVENT_LOGIN_LOCKED:
        ip = _as_text(payload.get("ip_address")) or "unknown"
        failed = _as_text(payload.get("failed_attempts"))
        max_att = _as_text(payload.get("max_attempts"))
        lock_min = _as_text(payload.get("lock_minutes"))
        occurred_at = _as_text(payload.get("occurred_at"))
        title = "Y2A-Auto 🚫 登录已被锁定"
        summary = f"{failed}/{max_att} | {lock_min} 分钟"
        body = _section_block(
            _kv("来源 IP", ip),
            _kv("失败次数", f"{failed}/{max_att}"),
            _kv("锁定时长", f"{lock_min} 分钟"),
            _kv("时间", occurred_at),
        )
        markdown = _markdown_lines("**🚫 登录已被锁定**", "", body)
        return NotificationMessage(title=title, summary=summary, markdown=markdown)

    if event_type in (EVENT_QR_LOGIN_SUCCESS, EVENT_QR_LOGIN_FAILED):
        platform = _as_text(payload.get("platform")) or "平台"
        is_success = event_type == EVENT_QR_LOGIN_SUCCESS
        icon = "✅" if is_success else "❌"
        status_text = "成功" if is_success else "失败"
        message = _truncate(_as_text(payload.get("message")) or ("Cookies 已保存" if is_success else "登录失败"), 300)
        occurred_at = _as_text(payload.get("occurred_at"))
        title = f"Y2A-Auto {icon} {platform}扫码登录{status_text}"
        summary = f"{platform} | {message}"
        body = _section_block(
            _kv("平台", platform),
            _kv("结果", message),
            _kv("时间", occurred_at),
        )
        markdown = _markdown_lines(f"**{icon} {platform}扫码登录{status_text}**", "", body)
        return NotificationMessage(title=title, summary=summary, markdown=markdown)

    if event_type in (EVENT_MONITOR_RUN_FINISHED, EVENT_MONITOR_ALERT):
        return _build_monitor_message(event_type, payload)

    title = "Y2A-Auto 💬 系统通知"
    serialized = json.dumps(payload, ensure_ascii=False)
    summary = _truncate(serialized, 180)
    body = _section_block(
        _kv("事件", event_type),
        _kv("内容", _truncate(serialized, 1500)),
    )
    markdown = _markdown_lines("**💬 系统通知**", "", body)
    return NotificationMessage(title=title, summary=summary, markdown=markdown)
