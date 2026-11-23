"""分组搜索（每行一组、分别搜索）、AI 选题大候选池不卡住、频道历史搬运配合 AI 选题的测试。"""

import sqlite3
import unittest
from unittest.mock import patch

from lxml import html

import app as web_app
from modules.youtube_monitor import (
    MAX_SEARCH_QUERY_GROUPS,
    YouTubeMonitor,
    _interleave_unique,
    normalize_search_queries,
    parse_search_queries,
)
from tests.test_monitor_ai_curation import _MonitorTestBase, _raw_video


class SearchQueryParsingTests(unittest.TestCase):
    def test_lines_are_trimmed_deduplicated_and_capped(self):
        text = "  root canal  treatment \r\n\r\nRoot Canal Treatment\ndental implant\n" + "\n".join(
            f"query {i}" for i in range(20))
        queries = parse_search_queries(text)
        self.assertEqual(queries[:2], ["root canal treatment", "dental implant"])
        self.assertEqual(len(queries), MAX_SEARCH_QUERY_GROUPS)
        self.assertEqual(normalize_search_queries(text), "\n".join(queries))

    def test_empty_values(self):
        for value in (None, "", "  \n \n"):
            with self.subTest(value=value):
                self.assertEqual(parse_search_queries(value), [])
                self.assertEqual(normalize_search_queries(value), "")

    def test_long_query_is_truncated(self):
        self.assertEqual(len(parse_search_queries("x" * 500)[0]), 200)

    def test_interleave_takes_turns_and_dedupes(self):
        self.assertEqual(
            _interleave_unique([["a1", "a2", "a3"], ["b1", "a1"], ["c1"]]),
            ["a1", "b1", "c1", "a2", "a3"],
        )
        self.assertEqual(_interleave_unique([]), [])


class _FakeResource:
    def __init__(self, kind, calls):
        self.kind, self.calls = kind, calls

    def list(self, **params):
        self.calls.append((self.kind, params))
        return (self.kind, params)


class _FakeYouTube:
    def __init__(self):
        self.calls = []

    def search(self):
        return _FakeResource("search.list", self.calls)

    def videos(self):
        return _FakeResource("videos.list", self.calls)


class GroupedSearchFetchTests(unittest.TestCase):
    def _monitor(self, remaining=None):
        monitor = YouTubeMonitor.__new__(YouTubeMonitor)
        monitor.youtube = _FakeYouTube()
        monitor._last_fetch_had_errors = False
        monitor._last_fetch_quota_skipped = False
        budget = iter(remaining) if remaining is not None else None
        monitor._quota_budget_remaining = (lambda: next(budget)) if budget else (lambda: 95)
        monitor.quota_calls = []
        monitor._try_consume_quota = lambda amount, config_id=None: monitor.quota_calls.append(amount) or True
        return monitor

    @staticmethod
    def _config(queries, max_results=5):
        return {
            "id": 7, "name": "evergreen", "keywords": "dental", "search_queries": queries,
            "max_results": max_results, "region_code": "US", "order_by": "viewCount",
            "video_types": "video,short", "category_id": "0",
        }

    @staticmethod
    def _executor(results, fail_queries=()):
        def execute(request, _operation, quota_consumer=None, **_kwargs):
            kind, params = request
            if kind == "search.list":
                if quota_consumer is not None:
                    quota_consumer()
                if params["q"] in fail_queries:
                    raise RuntimeError("backend error")
                return {"items": [{"id": {"videoId": vid}} for vid in results[params["q"]]]}
            return {"items": [{"id": vid} for vid in params["id"].split(",")]}
        return execute

    def test_each_line_is_searched_separately_and_merged(self):
        monitor = self._monitor()
        monitor._execute_with_retry = self._executor({
            "root canal": ["r1", "r2", "shared"],
            "dental implant": ["i1", "shared", "i2"],
        })
        videos = monitor._fetch_search_videos(
            self._config("root canal\ndental implant"), "2016-01-01T00:00:00Z")
        searches = [p for kind, p in monitor.youtube.calls if kind == "search.list"]
        self.assertEqual([p["q"] for p in searches], ["root canal", "dental implant"])
        self.assertTrue(all(p["maxResults"] == 10 and p["order"] == "viewCount" for p in searches))
        self.assertEqual(monitor.quota_calls, [1, 1])
        self.assertEqual([v["id"] for v in videos], ["r1", "i1", "r2", "shared", "i2"])
        self.assertFalse(monitor._last_fetch_had_errors)

    def test_details_are_fetched_in_batches_of_50(self):
        monitor = self._monitor()
        monitor._execute_with_retry = self._executor({
            "a": [f"a{i}" for i in range(50)],
            "b": [f"b{i}" for i in range(50)],
        })
        videos = monitor._fetch_search_videos(self._config("a\nb", max_results=25), "2016-01-01T00:00:00Z")
        detail_calls = [p for kind, p in monitor.youtube.calls if kind == "videos.list"]
        self.assertEqual([len(p["id"].split(",")) for p in detail_calls], [50, 50])
        self.assertEqual(len(videos), 100)

    def test_quota_running_out_stops_remaining_groups(self):
        monitor = self._monitor(remaining=[95, 1, 0])
        monitor._execute_with_retry = self._executor({"a": ["a1"], "b": ["b1"], "c": ["c1"]})
        videos = monitor._fetch_search_videos(self._config("a\nb\nc"), "2016-01-01T00:00:00Z")
        searches = [p["q"] for kind, p in monitor.youtube.calls if kind == "search.list"]
        self.assertEqual(searches, ["a", "b"])
        self.assertTrue(monitor._last_fetch_quota_skipped)
        self.assertEqual([v["id"] for v in videos], ["a1", "b1"])

    def test_one_failing_group_keeps_the_others(self):
        monitor = self._monitor()
        monitor._execute_with_retry = self._executor({"a": ["a1"], "b": ["b1"], "c": ["c1"]}, fail_queries={"b"})
        videos = monitor._fetch_search_videos(self._config("a\nb\nc"), "2016-01-01T00:00:00Z")
        self.assertEqual([v["id"] for v in videos], ["a1", "c1"])
        self.assertTrue(monitor._last_fetch_had_errors, "有一组失败，本轮算不完整，不推进 last_run_time")

    def test_without_groups_keywords_are_still_one_or_search(self):
        monitor = self._monitor()
        monitor._execute_with_retry = self._executor({"dental|teeth": ["v1"]})
        config = dict(self._config(""), keywords="dental teeth")
        videos = monitor._fetch_search_videos(config, "2016-01-01T00:00:00Z")
        searches = [p["q"] for kind, p in monitor.youtube.calls if kind == "search.list"]
        self.assertEqual(searches, ["dental|teeth"])
        self.assertEqual([v["id"] for v in videos], ["v1"])


class LargePoolCurationTests(_MonitorTestBase):
    """候选池远大于每轮评分上限时，打过分没达标的候选不能一直占着名额。"""

    def _run(self, config_id, videos, scorer):
        added = []

        def fake_add(_self, video_info, auto_start=True):
            added.append(video_info["id"])
            return f"task-{video_info['id']}"

        with patch.object(YouTubeMonitor, "_fetch_trending_videos", return_value=list(videos)), \
                patch("modules.youtube_monitor.score_videos", side_effect=scorer), \
                patch.object(YouTubeMonitor, "_add_video_to_tasks", autospec=True, side_effect=fake_add):
            ok, message = self.monitor.run_monitor(config_id)
        self.assertTrue(ok, message)
        return added, message

    def test_scoring_window_moves_past_rejected_candidates(self):
        videos = [_raw_video(f"v{i:03d}") for i in range(100)]
        good = {"v085"}
        calls = []

        def scorer(batch, criteria, keywords, app_config, logger=None):
            calls.append([v["id"] for v in batch])
            return {v["id"]: ((8, "讲解清楚") if v["id"] in good else (2, "信息少")) for v in batch}

        config_id = self._create_config(rate_limit_requests=3)
        first, _ = self._run(config_id, videos, scorer)
        second, _ = self._run(config_id, videos, scorer)
        third, message = self._run(config_id, videos, scorer)
        self.assertEqual([len(c) for c in calls], [40, 40, 20])
        self.assertEqual(calls[1][0], "v040")
        self.assertEqual(calls[2][0], "v080")
        self.assertEqual((first, second, third), ([], [], ["v085"]))
        self.assertIn("AI 选题：1/100", message)

        fourth, _ = self._run(config_id, videos, scorer)
        self.assertEqual(fourth, [])
        self.assertEqual(len(calls), 3, "全部打过分以后不应再调用模型")

    def test_cached_keepers_wait_for_later_runs_without_rescoring(self):
        videos = [_raw_video(f"k{i}") for i in range(5)]
        calls = []

        def scorer(batch, criteria, keywords, app_config, logger=None):
            calls.append([v["id"] for v in batch])
            return {v["id"]: (9 - int(v["id"][1:]), "好") for v in batch}

        config_id = self._create_config(rate_limit_requests=2)
        self.assertEqual(self._run(config_id, videos, scorer)[0], ["k0", "k1"])
        self.assertEqual(self._run(config_id, videos, scorer)[0], ["k2", "k3"])
        self.assertEqual(len(calls), 1)

    def test_grouped_queries_are_the_keyword_context_for_scoring(self):
        seen = []

        def scorer(batch, criteria, keywords, app_config, logger=None):
            seen.append(keywords)
            return {v["id"]: (7, "好") for v in batch}

        config_id = self._create_config(search_queries="root canal\ndental implant")
        self._run(config_id, [_raw_video("a")], scorer)
        self.assertEqual(seen, ["root canal；dental implant"])

    def test_historical_mode_with_ai_does_not_skip_by_offset(self):
        videos = [_raw_video(v) for v in ("old1", "old2", "new1")]
        scores = {"old1": (9, "好"), "old2": (8, "好"), "new1": (3, "差")}

        def scorer(batch, criteria, keywords, app_config, logger=None):
            return {v["id"]: scores[v["id"]] for v in batch}

        config_id = self._create_config(channel_mode="historical", rate_limit_requests=1)
        with sqlite3.connect(self.monitor.db_path) as conn:
            conn.execute("UPDATE monitor_configs SET historical_offset = 2 WHERE id = ?", (config_id,))
        added, _ = self._run(config_id, videos, scorer)
        self.assertEqual(added, ["old1"])

    def test_update_without_search_queries_keeps_them(self):
        config_id = self._create_config(search_queries="root canal\ndental implant")
        config = self.monitor.get_monitor_config(config_id)
        config.pop("search_queries")
        self.monitor.update_monitor_config(config_id, config)
        self.assertEqual(self.monitor.get_monitor_config(config_id)["search_queries"], "root canal\ndental implant")


class GroupedSearchFormTests(unittest.TestCase):
    def test_textarea_renders_saved_queries(self):
        config = {
            "name": "evergreen", "search_queries": "root canal\ndental implant", "channel_ids": "",
            "schedule_interval": 1440, "auto_add_to_tasks": True, "video_types": "video,short",
            "time_period": 3650,
        }
        with web_app.app.test_request_context("/youtube_monitor/config/1/edit"):
            rendered = web_app.render_template("youtube_monitor_config.html", config=config, is_edit=True)
        document = html.fromstring(rendered)
        self.assertEqual(document.get_element_by_id("search_queries").text_content(), "root canal\ndental implant")
        self.assertEqual(document.get_element_by_id("time_period").get("max"), "3650")

    def test_form_parser_normalizes_queries(self):
        from werkzeug.datastructures import MultiDict
        parsed = web_app._parse_monitor_extra_fields(MultiDict({"search_queries": " a \r\n\r\nA\nb "}))
        self.assertEqual(parsed["search_queries"], "a\nb")


if __name__ == "__main__":
    unittest.main()
