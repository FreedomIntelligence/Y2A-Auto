"""多个监控搜到同一个视频时只建一个任务：按其他监控的历史和任务表（含手动添加）去重。"""

import os
import tempfile
import unittest
from unittest.mock import patch

import modules.task_manager as tm
from modules.youtube_monitor import YouTubeMonitor
from tests.test_monitor_ai_curation import _MonitorTestBase, _raw_video

# YouTube 视频 ID 都是 11 位
A, B, C, D = "AAAAAAAAAAA", "BBBBBBBBBBB", "CCCCCCCCCCC", "DDDDDDDDDDD"


class CrossMonitorDedupeTests(_MonitorTestBase):
    def setUp(self):
        super().setUp()
        tm.init_db()  # 基类已把任务库指到临时目录

    def _run(self, config_id, videos):
        added = []

        def fake_add(_self, video_info, auto_start=True):
            added.append(video_info["id"])
            return f"task-{video_info['id']}"

        with patch.object(YouTubeMonitor, "_fetch_trending_videos", return_value=list(videos)), \
                patch.object(YouTubeMonitor, "_add_video_to_tasks", autospec=True, side_effect=fake_add):
            ok, message = self.monitor.run_monitor(config_id)
        self.assertTrue(ok, message)
        return added

    def _insert_task(self, url):
        conn = tm.get_db_connection()
        try:
            conn.execute(
                "INSERT INTO tasks (id, youtube_url, status) VALUES (?, ?, ?)",
                (f"task-{len(url)}-{url[-11:]}", url, tm.TASK_STATES["COMPLETED"]),
            )
            conn.commit()
        finally:
            conn.close()

    def test_video_added_by_another_monitor_is_skipped(self):
        first = self._create_config(name="m1", ai_filter_enabled=False, rate_limit_requests=5)
        second = self._create_config(name="m2", ai_filter_enabled=False, rate_limit_requests=5)
        self.assertEqual(self._run(first, [_raw_video(A)]), [A])
        self.assertEqual(self._run(second, [_raw_video(A), _raw_video(B)]), [B])

    def test_own_earlier_tasks_are_not_reported_as_reposted_elsewhere(self):
        config_id = self._create_config(ai_filter_enabled=False, rate_limit_requests=5)
        self.assertEqual(self._run(config_id, [_raw_video(A)]), [A])
        self._insert_task(f"https://www.youtube.com/watch?v={A}")  # 本监控自己加的任务
        self._insert_task(f"https://youtu.be/{B}")  # 手动加的任务
        with self.assertLogs("Y2A-Auto.YouTube-Monitor", level="INFO") as logs:
            self.assertEqual(self._run(config_id, [_raw_video(A), _raw_video(B), _raw_video(C)]), [C])
        skipped = [line for line in logs.output if "已由其他监控或手动加入过任务" in line]
        self.assertEqual(len(skipped), 1)
        self.assertIn("跳过 1 个", skipped[0])

    def test_video_only_seen_by_another_monitor_can_still_be_added(self):
        first = self._create_config(name="m1", ai_filter_enabled=False, auto_add_to_tasks=False)
        second = self._create_config(name="m2", ai_filter_enabled=False, rate_limit_requests=5)
        self.assertEqual(self._run(first, [_raw_video(A)]), [])
        self.assertEqual(self._run(second, [_raw_video(A)]), [A])

    def test_videos_already_in_the_task_list_are_skipped(self):
        self._insert_task(f"https://youtu.be/{A}")
        self._insert_task(f"https://www.youtube.com/shorts/{B}")
        self._insert_task(f"https://www.youtube.com/watch?v={C}&t=5s")
        config_id = self._create_config(ai_filter_enabled=False, rate_limit_requests=5)
        self.assertEqual(self._run(config_id, [_raw_video(v) for v in (A, B, C, D)]), [D])

    def test_reposted_videos_are_not_sent_for_ai_scoring(self):
        self._insert_task(f"https://www.youtube.com/watch?v={A}")
        scored = []

        def scorer(batch, criteria, keywords, app_config, logger=None):
            scored.extend(v["id"] for v in batch)
            return {v["id"]: (8, "讲解清楚") for v in batch}

        config_id = self._create_config(rate_limit_requests=5)
        with patch("modules.youtube_monitor.score_videos", side_effect=scorer):
            added = self._run(config_id, [_raw_video(A), _raw_video(B)])
        self.assertEqual(scored, [B])
        self.assertEqual(added, [B])


class ExistingVideoIdsTests(unittest.TestCase):
    def test_missing_task_db_returns_nothing_and_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tasks.db")
            with patch.object(tm, "DB_PATH", path):
                self.assertEqual(tm.existing_youtube_video_ids([A]), set())
            self.assertFalse(os.path.exists(path))

    def test_only_requested_ids_are_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(tm, "DB_PATH", os.path.join(tmp, "tasks.db")):
                tm.init_db()
                conn = tm.get_db_connection()
                try:
                    for i, url in enumerate((f"https://www.youtube.com/watch?v={A}", f"https://youtu.be/{B}")):
                        conn.execute("INSERT INTO tasks (id, youtube_url, status) VALUES (?, ?, ?)",
                                     (f"t{i}", url, tm.TASK_STATES["FAILED"]))
                    conn.commit()
                finally:
                    conn.close()
                self.assertEqual(tm.existing_youtube_video_ids([A, C]), {A})
                self.assertEqual(tm.existing_youtube_video_ids([]), set())


if __name__ == "__main__":
    unittest.main()
