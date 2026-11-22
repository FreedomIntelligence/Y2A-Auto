from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


EVENT_TASK_ADDED = "TASK_ADDED"
EVENT_TASK_COMPLETED = "TASK_COMPLETED"
EVENT_TASK_FAILED = "TASK_FAILED"
EVENT_LOGIN_SUCCESS = "LOGIN_SUCCESS"
EVENT_LOGIN_LOCKED = "LOGIN_LOCKED"
EVENT_QR_LOGIN_SUCCESS = "QR_LOGIN_SUCCESS"
EVENT_QR_LOGIN_FAILED = "QR_LOGIN_FAILED"
# 监控每轮运行的摘要（搜到多少、AI 保留多少、加了哪些任务）
EVENT_MONITOR_RUN_FINISHED = "MONITOR_RUN_FINISHED"
# 监控异常：运行失败、AI 选题失败、搜索 0 结果、配额跳过
EVENT_MONITOR_ALERT = "MONITOR_ALERT"

ALL_EVENT_TYPES = (
    EVENT_TASK_ADDED,
    EVENT_TASK_COMPLETED,
    EVENT_TASK_FAILED,
    EVENT_LOGIN_SUCCESS,
    EVENT_LOGIN_LOCKED,
    EVENT_QR_LOGIN_SUCCESS,
    EVENT_QR_LOGIN_FAILED,
    EVENT_MONITOR_RUN_FINISHED,
    EVENT_MONITOR_ALERT,
)

EVENT_CONFIG_KEY_MAP = {
    EVENT_TASK_ADDED: "NOTIFY_EVENT_TASK_ADDED",
    EVENT_TASK_COMPLETED: "NOTIFY_EVENT_TASK_COMPLETED",
    EVENT_TASK_FAILED: "NOTIFY_EVENT_TASK_FAILED",
    EVENT_LOGIN_SUCCESS: "NOTIFY_EVENT_LOGIN_SUCCESS",
    EVENT_LOGIN_LOCKED: "NOTIFY_EVENT_LOGIN_LOCKED",
    EVENT_QR_LOGIN_SUCCESS: "NOTIFY_EVENT_QR_LOGIN_SUCCESS",
    EVENT_QR_LOGIN_FAILED: "NOTIFY_EVENT_QR_LOGIN_FAILED",
    EVENT_MONITOR_RUN_FINISHED: "NOTIFY_EVENT_MONITOR_RUN_FINISHED",
    EVENT_MONITOR_ALERT: "NOTIFY_EVENT_MONITOR_ALERT",
}


def current_local_time_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True)
class NotificationEvent:
    event_type: str
    payload: dict[str, Any]
    occurred_at: str = field(default_factory=current_local_time_text)

    def as_payload(self) -> dict[str, Any]:
        payload = dict(self.payload or {})
        payload.setdefault("occurred_at", self.occurred_at)
        return payload


@dataclass(frozen=True)
class NotificationMessage:
    title: str
    summary: str
    markdown: str
