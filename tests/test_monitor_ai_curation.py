"""监控 AI 选题、对齐调度、监控通知、颜文字标题过滤的测试。"""

import os
import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from modules import monitor_curation as mc
from modules.ai_enhancer import has_enough_title_text
from modules.notifications import (
    EVENT_MONITOR_ALERT,
    EVENT_MONITOR_RUN_FINISHED,
    NotificationEvent,
    build_notification_message,
)
from modules.youtube_monitor import (
    YouTubeMonitor,
    missed_scheduled_run,
    normalize_schedule_time,
    parse_schedule_time,
    schedule_anchor,
)


def _raw_video(vid, title=None, views=10000, duration="PT45S", description="", tags=None):
    """能通过 _filter_videos 的原始 YouTube API 视频字典。"""
    return {
        "id": vid,
        "snippet": {
            "title": title or f"How dentists fix cavity {vid}",
            "channelTitle": f"Channel {vid}",
            "channelId": f"UC{vid}",
            "publishedAt": "2026-11-20T00:00:00Z",
            "liveBroadcastContent": "none",
            "description": description,
            "tags": tags or ["dental"],
        },
        "contentDetails": {"duration": duration},
        "statistics": {"viewCount": str(views), "likeCount": "100", "commentCount": "5"},
    }


APP_CONFIG = {
    "OPENAI_API_KEY": "sk-test",
    "OPENAI_BASE_URL": "https://api.example.com",
    "OPENAI_MODEL_NAME": "model-a",
    "TRANSLATE_TITLE": True,
    "MONITOR_SKIP_UNTRANSLATABLE_TITLES": True,
    "NOTIFY_ENABLED": False,
}


class CurationHelpersTests(unittest.TestCase):
    def test_min_score_is_clamped_and_defaults(self):
        self.assertEqual(mc.normalize_min_score("7"), 7)
        self.assertEqual(mc.normalize_min_score(12), 10)
        self.assertEqual(mc.normalize_min_score(-3), 0)
        self.assertEqual(mc.normalize_min_score("6.6"), 7)
        for bad in (None, "", "abc", float("nan")):
            with self.subTest(value=bad):
                self.assertEqual(mc.normalize_min_score(bad), mc.DEFAULT_AI_FILTER_MIN_SCORE)

    def test_empty_criteria_falls_back_to_default(self):
        self.assertEqual(mc.effective_criteria("   "), mc.DEFAULT_AI_FILTER_CRITERIA)
        self.assertEqual(mc.effective_criteria("只要补牙过程"), "只要补牙过程")
        self.assertEqual(len(mc.normalize_criteria("x" * 5000)), mc.AI_CURATION_CRITERIA_MAX_LENGTH)

    def test_fingerprint_changes_with_criteria_keywords_and_model(self):
        base = mc.verdict_fingerprint("标准A", "dental", "model-a")
        self.assertEqual(base, mc.verdict_fingerprint("标准A", "dental", "model-a"))
        self.assertNotEqual(base, mc.verdict_fingerprint("标准B", "dental", "model-a"))
        self.assertNotEqual(base, mc.verdict_fingerprint("标准A", "teeth", "model-a"))
        self.assertNotEqual(base, mc.verdict_fingerprint("标准A", "dental", "model-b"))

    def test_payload_is_compact(self):
        video = {
            "id": "v1", "title": "T", "channel_title": "C", "video_type": "short",
            "duration": "PT1M5S", "view_count": 12, "like_count": 3, "comment_count": 1,
            "published_at": "2026-11-20T08:00:00Z",
            "description": "Book now https://clinic.example.com/x  " + "word " * 200,
            "tags": [f"t{i}" for i in range(30)],
        }
        item = mc.build_candidate_payload([video])[0]
        self.assertEqual(item["duration_s"], 65)
        self.assertEqual(item["published"], "2026-11-20")
        self.assertNotIn("https://", item["description"])
        self.assertLessEqual(len(item["description"]), 400)
        self.assertEqual(len(item["tags"]), 12)

    def test_parse_verdicts_keeps_only_requested_ids(self):
        parsed = {"results": [
            {"id": "a", "score": 8, "reason": "讲解清楚"},
            {"id": "b", "score": "3.4", "reason": "广告"},
            {"id": "zzz", "score": 10, "reason": "不在输入里"},
            {"id": "a", "score": 0, "reason": "重复项忽略"},
            {"id": "c", "score": "高", "reason": "分数无效"},
            "garbage",
        ]}
        verdicts = mc.parse_verdicts(parsed, ["a", "b", "c"])
        self.assertEqual(verdicts, {"a": (8, "讲解清楚"), "b": (3, "广告")})

    def test_parse_verdicts_rejects_unusable_output(self):
        for parsed in (None, {}, {"results": "x"}, {"results": [{"id": "x", "score": 5}]}):
            with self.subTest(parsed=parsed):
                with self.assertRaises(mc.AICurationError):
                    mc.parse_verdicts(parsed, ["a"])

    def test_rank_orders_by_score_then_original_order(self):
        videos = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}, {"id": "e"}]
        verdicts = {"a": (6, ""), "b": (9, ""), "c": (2, "广告"), "d": (9, "")}
        kept, rejected, unscored = mc.rank_by_verdicts(videos, verdicts, 6)
        self.assertEqual([v["id"] for v in kept], ["b", "d", "a"])
        self.assertEqual([v["id"] for v in rejected], ["c"])
        self.assertEqual([v["id"] for v in unscored], ["e"])
        self.assertEqual(rejected[0]["ai_reason"], "广告")


class ScoreVideosTests(unittest.TestCase):
    VIDEOS = [{"id": "a", "title": "A"}, {"id": "b", "title": "B"}]

    def test_success_returns_verdicts(self):
        calls = {}

        def fake_request(client, model, system_prompt, payload, **kwargs):
            calls["model"] = model
            calls["prompt"] = system_prompt
            calls["ids"] = [v["id"] for v in payload["videos"]]
            return {"results": [{"id": "a", "score": 9, "reason": "好"}, {"id": "b", "score": 1, "reason": "差"}]}

        with patch("modules.ai_enhancer.get_openai_client", return_value=object()), \
                patch("modules.ai_enhancer._request_json_object", side_effect=fake_request):
            verdicts = mc.score_videos(self.VIDEOS, "只要牙科科普", "dental", APP_CONFIG)
        self.assertEqual(verdicts, {"a": (9, "好"), "b": (1, "差")})
        self.assertEqual(calls["model"], "model-a")
        self.assertEqual(calls["ids"], ["a", "b"])
        self.assertIn("只要牙科科普", calls["prompt"])
        self.assertIn("dental", calls["prompt"])

    def test_failures_raise_curation_error(self):
        with patch("modules.ai_enhancer.get_openai_client", return_value=object()), \
                patch("modules.ai_enhancer._request_json_object", side_effect=TimeoutError("read timeout")):
            with self.assertRaises(mc.AICurationError):
                mc.score_videos(self.VIDEOS, "", "", APP_CONFIG)
        with patch("modules.ai_enhancer.get_openai_client", return_value=object()), \
                patch("modules.ai_enhancer._request_json_object", return_value=None):
            with self.assertRaises(mc.AICurationError):
                mc.score_videos(self.VIDEOS, "", "", APP_CONFIG)
        with self.assertRaises(mc.AICurationError):
            mc.score_videos(self.VIDEOS, "", "", {**APP_CONFIG, "OPENAI_API_KEY": ""})

    def test_no_videos_skips_model(self):
        with patch("modules.ai_enhancer._request_json_object") as fake:
            self.assertEqual(mc.score_videos([], "", "", APP_CONFIG), {})
        fake.assert_not_called()


class EmoticonTitleTests(unittest.TestCase):
    def test_emoticon_only_titles_are_not_enough(self):
        for title in ("シ", "🦷🎀🦷 シ", "シ # #", "ㅋㅋㅋ #치과", "#dental #shorts", "😂😂 #dentista", "牙"):
            with self.subTest(title=title):
                self.assertFalse(has_enough_title_text(title))

    def test_real_titles_pass(self):
        for title in ("Crown 🔥 #dental", "Root Canal treatment🦷#dental", "缝合 💯💯",
                      "Сколько стоит лечить зубы", "അങ്ങനെ വീണ്ടും പോയി", "OK"):
            with self.subTest(title=title):
                self.assertTrue(has_enough_title_text(title))

    def test_monitor_skips_emoticon_title(self):
        monitor = object.__new__(YouTubeMonitor)
        config = {
            "video_types": "video,short", "min_view_count": 0, "min_like_count": 0,
            "min_comment_count": 0, "exclude_keywords": "", "title_keywords": "",
            "exclude_channel_ids": "", "channel_ids": "", "min_duration": 0, "max_duration": 0,
        }
        info = {"id": "x", "title": "🦷🎀🦷 シ", "channel_id": "UC1", "duration": "PT20S",
                "view_count": 1, "like_count": 0, "comment_count": 0, "video_type": "short",
                "published_at": "2026-11-20T00:00:00Z"}
        self.assertFalse(monitor._meets_criteria(info, config, skip_untranslatable_titles=True))
        self.assertTrue(monitor._meets_criteria(info, config, skip_untranslatable_titles=False))


class _MonitorTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patchers = [
            patch("modules.youtube_monitor.get_app_subdir", side_effect=self._get_app_subdir),
            patch.object(YouTubeMonitor, "_init_youtube_api", return_value=(False, "missing")),
            patch("modules.youtube_monitor.load_config", side_effect=lambda: dict(self.app_config)),
        ]
        self.app_config = dict(APP_CONFIG)
        for p in self.patchers:
            p.start()
        self.monitor = YouTubeMonitor()
        self.monitor.youtube = object()

    def tearDown(self):
        try:
            if getattr(self.monitor.scheduler, "running", False):
                self.monitor.scheduler.shutdown(wait=False)
        finally:
            for p in reversed(self.patchers):
                p.stop()
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _get_app_subdir(self, name):
        path = os.path.join(self.tmpdir, name)
        os.makedirs(path, exist_ok=True)
        return path

    def _create_config(self, **overrides):
        data = {
            "name": "dental", "keywords": "dental", "max_results": 20, "rate_limit_requests": 2,
            "auto_add_to_tasks": True, "schedule_type": "manual",
            "ai_filter_enabled": True, "ai_filter_criteria": "只要牙科科普", "ai_filter_min_score": 6,
        }
        data.update(overrides)
        return self.monitor.create_monitor_config(data)


class RunMonitorCurationTests(_MonitorTestBase):
    CANDIDATES = [_raw_video(v) for v in ("ad", "good1", "meh", "good2", "best")]
    SCORES = {"ad": (0, "诊所广告"), "good1": (7, "补牙过程"), "meh": (5, "信息少"),
              "good2": (7, "洗牙"), "best": (9, "讲解清楚")}

    def _run(self, config_id, scorer):
        added = []

        def fake_add(_self, video_info, auto_start=True):
            added.append(video_info["id"])
            return f"task-{video_info['id']}"

        with patch.object(YouTubeMonitor, "_fetch_trending_videos", return_value=list(self.CANDIDATES)), \
                patch("modules.youtube_monitor.score_videos", side_effect=scorer), \
                patch.object(YouTubeMonitor, "_add_video_to_tasks", autospec=True, side_effect=fake_add):
            ok, message = self.monitor.run_monitor(config_id)
        return ok, message, added

    def _scorer(self, calls):
        def scorer(videos, criteria, keywords, app_config, logger=None):
            calls.append([v["id"] for v in videos])
            return {v["id"]: self.SCORES[v["id"]] for v in videos}
        return scorer

    def test_adds_highest_scores_first_and_records_score(self):
        config_id = self._create_config()
        calls = []
        ok, message, added = self._run(config_id, self._scorer(calls))
        self.assertTrue(ok, message)
        self.assertEqual(added, ["best", "good1"])  # 9 分在前，两个 7 分按原顺序，上限 2 个
        self.assertEqual(calls, [["ad", "good1", "meh", "good2", "best"]])
        self.assertIn("AI 选题：3/5", message)
        with sqlite3.connect(self.monitor.db_path) as conn:
            rows = dict(conn.execute(
                "SELECT video_id, ai_score FROM monitor_history WHERE config_id = ?", (config_id,)).fetchall())
        # 被淘汰的和超出上限的都不写进历史，下轮仍是候选
        self.assertEqual(rows, {"best": 9, "good1": 7})

    def test_second_run_uses_cache_and_skips_processed(self):
        config_id = self._create_config()
        calls = []
        self._run(config_id, self._scorer(calls))
        ok, _, added = self._run(config_id, self._scorer(calls))
        self.assertTrue(ok)
        self.assertEqual(added, ["good2"])  # 剩下唯一达标的
        self.assertEqual(len(calls), 1, "第二轮全部命中评分缓存，不应再调用模型")

    def test_changing_criteria_rescores(self):
        config_id = self._create_config()
        calls = []
        self._run(config_id, self._scorer(calls))
        config = self.monitor.get_monitor_config(config_id)
        config["ai_filter_criteria"] = "只要正畸"
        self.monitor.update_monitor_config(config_id, config)
        self._run(config_id, self._scorer(calls))
        self.assertEqual(len(calls), 2)
        self.assertEqual(sorted(calls[1]), ["ad", "good2", "meh"])  # 已入队的两个不再打分

    def test_model_failure_adds_nothing_and_keeps_cursor(self):
        config_id = self._create_config()

        def failing(*args, **kwargs):
            raise mc.AICurationError("模型超时")

        ok, message, added = self._run(config_id, failing)
        self.assertFalse(ok)
        self.assertIn("AI 选题失败", message)
        self.assertEqual(added, [])
        self.assertIsNone(self.monitor.get_monitor_config(config_id)["last_run_time"])
        with sqlite3.connect(self.monitor.db_path) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM monitor_history").fetchone()[0], 0)

    def test_disabled_keeps_original_behaviour(self):
        config_id = self._create_config(ai_filter_enabled=False)
        calls = []
        ok, _, added = self._run(config_id, self._scorer(calls))
        self.assertTrue(ok)
        self.assertEqual(added, ["ad", "good1"])
        self.assertEqual(calls, [])

    def test_update_without_ai_fields_keeps_them(self):
        config_id = self._create_config(schedule_time="17:00")
        config = self.monitor.get_monitor_config(config_id)
        for key in ("ai_filter_enabled", "ai_filter_criteria", "ai_filter_min_score", "schedule_time"):
            config.pop(key)
        self.monitor.update_monitor_config(config_id, config)
        saved = self.monitor.get_monitor_config(config_id)
        self.assertTrue(saved["ai_filter_enabled"])
        self.assertEqual(saved["ai_filter_criteria"], "只要牙科科普")
        self.assertEqual(saved["ai_filter_min_score"], 6)
        self.assertEqual(saved["schedule_time"], "17:00")


class MonitorNotificationTests(_MonitorTestBase):
    def _run_capture(self, config_id, videos, app_overrides=None):
        self.app_config.update({"NOTIFY_ENABLED": True, "NOTIFY_EVENT_MONITOR_RUN_FINISHED": True,
                                "NOTIFY_EVENT_MONITOR_ALERT": True, **(app_overrides or {})})
        events = []
        with patch.object(YouTubeMonitor, "_fetch_trending_videos", return_value=videos), \
                patch("modules.youtube_monitor.score_videos",
                      side_effect=lambda vs, *a, **k: {v["id"]: (8, "好") for v in vs}), \
                patch.object(YouTubeMonitor, "_add_video_to_tasks", return_value="task-1"), \
                patch("modules.youtube_monitor.emit_notification_event", side_effect=events.append):
            ok, message = self.monitor.run_monitor(config_id)
        return ok, message, events

    def test_summary_event_lists_added_videos(self):
        config_id = self._create_config()
        ok, _, events = self._run_capture(config_id, [_raw_video("a"), _raw_video("b")])
        self.assertTrue(ok)
        self.assertEqual([e.event_type for e in events], [EVENT_MONITOR_RUN_FINISHED])
        payload = events[0].payload
        self.assertEqual(payload["added_count"], 2)
        self.assertEqual(payload["ai_kept"], 2)
        message = build_notification_message(events[0])
        self.assertIn("监控运行完成", message.title)
        self.assertIn("[8分]", message.markdown)
        self.assertIn("加入任务队列：**2 个", message.markdown.replace("** ", "**"))

    def test_zero_results_sends_alert_only(self):
        config_id = self._create_config()
        ok, _, events = self._run_capture(config_id, [])
        self.assertTrue(ok)
        self.assertEqual([e.event_type for e in events], [EVENT_MONITOR_ALERT])
        self.assertIn("没有返回任何视频", events[0].payload["alert_reason"])

    def test_everything_filtered_out_sends_alert(self):
        """11-17、11-20 的情况：搜到视频，但全被过滤掉，当天没有新任务。"""
        config_id = self._create_config(ai_filter_enabled=False, video_types="video,short")
        live_replays = [_raw_video("a"), _raw_video("b")]
        for video in live_replays:
            video["liveStreamingDetails"] = {"actualEndTime": "2026-11-21T00:00:00Z"}
        ok, _, events = self._run_capture(config_id, live_replays)
        self.assertTrue(ok)
        self.assertEqual([e.event_type for e in events], [EVENT_MONITOR_ALERT])
        reason = events[0].payload["alert_reason"]
        self.assertIn("本轮没有加入任何任务", reason)
        self.assertIn("搜索 2 个，规则过滤后 0 个", reason)

    def test_nothing_new_without_auto_add_is_not_an_alert(self):
        config_id = self._create_config(auto_add_to_tasks=False)
        ok, _, events = self._run_capture(config_id, [_raw_video("a")])
        self.assertTrue(ok)
        self.assertEqual([e.event_type for e in events], [EVENT_MONITOR_RUN_FINISHED])

    def test_zero_results_on_channel_monitor_is_not_an_alert(self):
        config_id = self._create_config(channel_ids="UCabc")
        ok, _, events = self._run_capture(config_id, [])
        self.assertTrue(ok)
        self.assertEqual([e.event_type for e in events], [EVENT_MONITOR_RUN_FINISHED])

    def test_ai_failure_sends_alert(self):
        config_id = self._create_config()
        self.app_config.update({"NOTIFY_ENABLED": True, "NOTIFY_EVENT_MONITOR_ALERT": True})
        events = []
        with patch.object(YouTubeMonitor, "_fetch_trending_videos", return_value=[_raw_video("a")]), \
                patch("modules.youtube_monitor.score_videos", side_effect=mc.AICurationError("超时")), \
                patch("modules.youtube_monitor.emit_notification_event", side_effect=events.append):
            ok, _ = self.monitor.run_monitor(config_id)
        self.assertFalse(ok)
        self.assertEqual([e.event_type for e in events], [EVENT_MONITOR_ALERT])
        self.assertEqual(events[0].payload["ai_error"], "超时")
        rendered = build_notification_message(events[0])
        self.assertIn("监控异常", rendered.title)
        self.assertIn("失败：超时", rendered.markdown)

    def test_event_toggle_off_sends_nothing(self):
        config_id = self._create_config()
        _, _, events = self._run_capture(
            config_id, [_raw_video("a")], {"NOTIFY_EVENT_MONITOR_RUN_FINISHED": False})
        self.assertEqual(events, [])


class ScheduleAnchorTests(_MonitorTestBase):
    def test_parse_and_normalize(self):
        self.assertEqual(parse_schedule_time("17:00"), (17, 0))
        self.assertEqual(parse_schedule_time(" 7:05 "), (7, 5))
        self.assertEqual(parse_schedule_time("08：30"), (8, 30))
        for bad in ("", None, "24:00", "12:60", "abc", "1700"):
            with self.subTest(value=bad):
                self.assertIsNone(parse_schedule_time(bad))
        self.assertEqual(normalize_schedule_time("7:05"), "07:05")
        self.assertEqual(normalize_schedule_time("nope"), "")

    def test_anchor_is_today_at_time(self):
        now = datetime(2026, 11, 22, 16, 41, 13)
        self.assertEqual(schedule_anchor(17, 0, now), datetime(2026, 11, 22, 17, 0, 0))

    def test_missed_run_detection(self):
        now = datetime(2026, 11, 22, 9, 0, 0)  # UTC
        self.assertTrue(missed_scheduled_run(None, 1440, now))
        self.assertTrue(missed_scheduled_run("garbage", 1440, now))
        self.assertFalse(missed_scheduled_run("2026-11-21 09:30:00", 1440, now))
        self.assertTrue(missed_scheduled_run("2026-11-21 08:00:00", 1440, now))

    def _jobs(self):
        return {job.id: job for job in self.monitor.scheduler.get_jobs()}

    def test_anchored_schedule_fires_at_time_regardless_of_save_time(self):
        config_id = self._create_config(schedule_type="auto", schedule_interval=1440, schedule_time="17:00")
        job = self._jobs()[f"monitor_{config_id}"]
        next_run = job.next_run_time if hasattr(job, "next_run_time") else job.trigger.get_next_fire_time(None, datetime.now(job.trigger.timezone))
        self.assertEqual((next_run.hour, next_run.minute), (17, 0))
        self.assertLessEqual(next_run.replace(tzinfo=None) - datetime.now(), timedelta(days=1))

    def test_startup_skips_immediate_run_when_not_missed(self):
        config_id = self._create_config(schedule_type="auto", schedule_interval=1440, schedule_time="17:00")
        self.monitor._remove_schedule(config_id)
        with sqlite3.connect(self.monitor.db_path) as conn:
            conn.execute("UPDATE monitor_configs SET last_run_time = datetime('now', '-2 hours') WHERE id = ?",
                         (config_id,))
        self.monitor.start_all_schedules(immediate=True)
        self.assertNotIn(f"monitor_immediate_{config_id}", self._jobs())
        self.assertIn(f"monitor_{config_id}", self._jobs())

    def test_startup_catches_up_missed_run(self):
        config_id = self._create_config(schedule_type="auto", schedule_interval=1440, schedule_time="17:00")
        self.monitor._remove_schedule(config_id)
        with sqlite3.connect(self.monitor.db_path) as conn:
            conn.execute("UPDATE monitor_configs SET last_run_time = datetime('now', '-30 hours') WHERE id = ?",
                         (config_id,))
        self.monitor.start_all_schedules(immediate=True)
        self.assertIn(f"monitor_immediate_{config_id}", self._jobs())

    def test_unanchored_schedule_keeps_immediate_run(self):
        config_id = self._create_config(schedule_type="auto", schedule_interval=1440)
        self.monitor._remove_schedule(config_id)
        with sqlite3.connect(self.monitor.db_path) as conn:
            conn.execute("UPDATE monitor_configs SET last_run_time = datetime('now', '-2 hours') WHERE id = ?",
                         (config_id,))
        self.monitor.start_all_schedules(immediate=True)
        self.assertIn(f"monitor_immediate_{config_id}", self._jobs())


class AlertRendererTests(unittest.TestCase):
    def test_alert_message_mentions_reason(self):
        event = NotificationEvent(EVENT_MONITOR_ALERT, {
            "config_name": "口腔健康科普", "alert_reason": "搜索没有返回任何视频",
            "fetched_count": 0, "filtered_count": 0, "added_count": 0,
        })
        message = build_notification_message(event)
        self.assertIn("监控异常", message.title)
        self.assertIn("口腔健康科普", message.summary)
        self.assertIn("搜索没有返回任何视频", message.markdown)


if __name__ == "__main__":
    unittest.main()
