"""无人值守搬运流水线加固（FreedomAI 部署中实际出现的问题）。

覆盖：
1. 非拉丁文字标题（俄语、马拉雅拉姆语等）不再被当成「预清洗后无可用内容」；
2. 只有话题标签 / 电话号码 / 链接的简介按「留空」处理，不再把任务卡进人工审核；
3. 监控阶段跳过只有话题标签的标题（这类标题必然无法自动翻译）；
4. bilibili 连续投稿的最小间隔，避免 137022「投稿过于频繁」；
5. 137022 与网络类上传失败自动延后重试，有次数上限；
6. 设置了期望 UID 时，登录的 bilibili 账号不一致就拒绝上传。

测试里的标题、简介和错误信息都取自线上任务的原文。
"""
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules import ai_enhancer as ai
from modules import task_manager as tm
from modules.bilibili_uploader import BilibiliUploader, check_expected_bilibili_account
from modules.youtube_monitor import YouTubeMonitor, should_skip_untranslatable_titles


ERR_137022 = "上传失败: bilibili上传异常: 接口返回错误代码：137022，信息：投稿过于频繁，请稍后再试"
ERR_CURL_28 = (
    "上传失败: bilibili上传异常: Failed to perform, curl: (28) Operation timed out after "
    "30001 milliseconds with 0 bytes received. See https://curl.se/libcurl/c/libcurl-errors.html "
    "first for more details."
)
ERR_CURL_60 = (
    "上传失败: bilibili上传异常: Failed to perform, curl: (60) SSL certificate problem: "
    "certificate has expired. See https://curl.se/libcurl/c/libcurl-errors.html first for more details."
)


class TranslatableContentTests(unittest.TestCase):
    def test_non_latin_titles_are_translatable(self):
        for title in (
            "Сколько стоит лечить зубы в Новой Зеландии? #newzealand #dentistry #clinicnewzealand",
            "അങ്ങനെ വീണ്ടും പോയി ഒരു പല്ല് എടുത്തു 😄#hospital #dental #dentalclinic",
            "दांत का दर्द #shorts",
            "السن خرج بالكامل بعد خبطة! 🦷",
            "앞니임플란트 #dentist",
            "根管治疗🦷",
        ):
            with self.subTest(title=title):
                self.assertTrue(ai.has_translatable_content(title, "title"))

    def test_hashtag_or_emoji_only_titles_are_not_translatable(self):
        for title in (
            "#Dentistry #funny #explortheworld #reelinstagram #worldexplorer #reels #comedy",
            "#dental#youtubeshorts #viralvideo #viral #explore #smile #subscribe #implant",
            "#love 🥰 #muhammadﷺ #@ApexDentalLabWork#my #dental #art🤲 #galez #shorts #viralshorts",
            "#foryou #fry #dentist #dental",
            "🦷✨🔥",
            "",
        ):
            with self.subTest(title=title):
                self.assertFalse(ai.has_translatable_content(title, "title"))

    def test_contact_or_link_only_descriptions_are_not_translatable(self):
        for description in (
            "☎️ 98.360-70-08",
            "#dentalconsultant #painlessdentistry #dentistrymarket #dentistryservices #dentalclinic",
            "https://example.com/booking",
        ):
            with self.subTest(description=description):
                self.assertFalse(ai.has_translatable_content(description, "description"))

    def test_regular_description_is_translatable(self):
        self.assertTrue(ai.has_translatable_content(
            "How is dentistry in New Zealand different? We show the inside of a clinic.", "description"))
        self.assertTrue(ai.has_translatable_content("Как лечат зубы в Новой Зеландии", "description"))

    def test_pre_clean_strips_glued_and_unicode_hashtags(self):
        cleaned = ai._pre_clean(
            "Dental Work 🦷✨ । #ceramic #crown #@ApexDentalLabWork #my #dental#art🤲#dentallab#shorts#support❤",
            content_type="title",
        )
        self.assertNotIn("#", cleaned)
        self.assertTrue(cleaned.startswith("Dental Work"))

    def test_pre_clean_keeps_language_names_with_hash(self):
        self.assertEqual(ai._pre_clean("C#编程入门 #shorts", content_type="title"), "C#编程入门")
        self.assertEqual(ai._pre_clean("F# vs C# 对比 #coding", content_type="title"), "F# vs C# 对比")
        self.assertEqual(ai._pre_clean("Root Canal treatment🦷#dental #shorts", content_type="title"),
                         "Root Canal treatment🦷")

    def test_pre_clean_keeps_cyrillic_text(self):
        cleaned = ai._pre_clean(
            "Сколько стоит лечить зубы в Новой Зеландии? #newzealand #dentistry", content_type="title")
        self.assertEqual(cleaned, "Сколько стоит лечить зубы в Новой Зеландии?")


class MetadataTranslationTests(unittest.TestCase):
    OPENAI_CONFIG = {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL_NAME": "test-model"}

    def _translate(self, title, description, model_output, **kwargs):
        with patch.object(ai, "get_openai_client", return_value=object()), \
                patch.object(ai, "_request_json_object", return_value=model_output) as request:
            result = ai.translate_video_metadata(
                title, description, openai_config=self.OPENAI_CONFIG, task_id="test-hardening", **kwargs)
        return result, request

    def test_cyrillic_title_is_sent_to_the_model(self):
        result, request = self._translate(
            "Сколько стоит лечить зубы в Новой Зеландии? #newzealand #dentistry",
            "",
            {"title": "在新西兰看牙要花多少钱？"},
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(result["title"], "在新西兰看牙要花多少钱？")
        self.assertEqual(request.call_args.kwargs["payload"]["title"], "Сколько стоит лечить зубы в Новой Зеландии?")

    def test_hashtag_only_description_is_skipped_not_failed(self):
        result, request = self._translate(
            "Teeth scaling with no pain #dentalscaling #dentalcare",
            "#dentalconsultant #painlessdentistry #dentistrymarket #dentistryservices #dentalclinic",
            {"title": "无痛洁牙：洗牙及抛光清洁"},
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(result["failed_fields"], {})
        self.assertEqual(result["skipped_fields"], ["description"])
        self.assertEqual(result["description"], "")
        self.assertNotIn("description", request.call_args.kwargs["payload"])

    def test_phone_number_description_is_skipped_not_failed(self):
        result, _ = self._translate(
            "Ceramic crown 🔥 #dental#dentist#dentistry#shorts", "☎️ 98.360-70-08", {"title": "陶瓷冠"})
        self.assertTrue(result["success"], result)
        self.assertEqual(result["skipped_fields"], ["description"])

    def test_hashtag_only_title_still_goes_to_manual_review(self):
        result, request = self._translate("#foryou #fry #dentist #dental", "", {"title": "不应被调用"})
        self.assertFalse(result["success"])
        self.assertEqual(result["failed_fields"], {"title": ["no_meaningful_content_after_preclean"]})
        request.assert_not_called()


class TranslationRequirementTests(unittest.TestCase):
    CONFIG = {"TRANSLATE_TITLE": True, "TRANSLATE_DESCRIPTION": True}

    def test_untranslatable_description_is_not_required(self):
        task = {
            "video_title_original": "Crown 🔥 #dental",
            "video_title_translated": "牙冠",
            "description_original": "☎️ 98.360-70-08",
            "description_translated": "",
        }
        self.assertEqual(tm._get_missing_required_translation_fields(task, self.CONFIG), [])

    def test_real_description_is_still_required(self):
        task = {
            "video_title_original": "Root canal cost",
            "video_title_translated": "根管治疗费用",
            "description_original": "We explain why root canal treatment costs differ between clinics.",
            "description_translated": "",
        }
        self.assertEqual(tm._get_missing_required_translation_fields(task, self.CONFIG), ["description"])

    def test_upload_description_never_falls_back_to_hashtag_only_original(self):
        hashtags_only = {"description_original": "#dental #smile #implant", "description_translated": ""}
        self.assertEqual(tm._resolve_upload_description(hashtags_only, self.CONFIG), "")
        translated = {"description_original": "#dental", "description_translated": "中文简介"}
        self.assertEqual(tm._resolve_upload_description(translated, self.CONFIG), "中文简介")
        real_text = {"description_original": "Real text here", "description_translated": ""}
        self.assertEqual(tm._resolve_upload_description(real_text, self.CONFIG), "Real text here")
        # 没开启简介翻译时，原样使用原简介（保持旧行为）
        self.assertEqual(
            tm._resolve_upload_description(hashtags_only, {"TRANSLATE_DESCRIPTION": False}),
            "#dental #smile #implant",
        )


class MonitorTitleFilterTests(unittest.TestCase):
    MONITOR_CONFIG = {
        "video_types": "video,short",
        "min_view_count": 0,
        "min_like_count": 0,
        "min_comment_count": 0,
        "exclude_keywords": "",
        "title_keywords": "dental,dentist,зубы",
        "exclude_channel_ids": "",
        "channel_ids": "",
        "min_duration": 0,
        "max_duration": 0,
    }

    def _info(self, title):
        return {
            "title": title,
            "view_count": 100,
            "like_count": 1,
            "comment_count": 0,
            "channel_id": "channel",
            "duration": "PT30S",
            "published_at": "2026-11-20T00:00:00Z",
            "video_type": "short",
        }

    def test_hashtag_only_title_is_skipped_only_when_enabled(self):
        monitor = object.__new__(YouTubeMonitor)
        info = self._info("#dental #dentallife #dentalmemes #dentalhumor #dentalassistant")
        self.assertFalse(monitor._meets_criteria(info, self.MONITOR_CONFIG, skip_untranslatable_titles=True))
        self.assertTrue(monitor._meets_criteria(info, self.MONITOR_CONFIG, skip_untranslatable_titles=False))
        self.assertTrue(monitor._meets_criteria(info, self.MONITOR_CONFIG))

    def test_non_latin_title_is_kept(self):
        monitor = object.__new__(YouTubeMonitor)
        info = self._info("Сколько стоит лечить зубы в Новой Зеландии? #dentistry")
        self.assertTrue(monitor._meets_criteria(info, self.MONITOR_CONFIG, skip_untranslatable_titles=True))

    def test_filter_only_applies_when_title_translation_is_on(self):
        self.assertTrue(should_skip_untranslatable_titles(
            {"MONITOR_SKIP_UNTRANSLATABLE_TITLES": True, "TRANSLATE_TITLE": True}))
        self.assertFalse(should_skip_untranslatable_titles(
            {"MONITOR_SKIP_UNTRANSLATABLE_TITLES": True, "TRANSLATE_TITLE": False}))
        self.assertFalse(should_skip_untranslatable_titles(
            {"MONITOR_SKIP_UNTRANSLATABLE_TITLES": False, "TRANSLATE_TITLE": True}))


class UploadRetryClassificationTests(unittest.TestCase):
    def test_rate_limit_and_network_errors_are_retryable(self):
        self.assertEqual(tm.classify_retryable_upload_error(ERR_137022), "rate_limit")
        self.assertEqual(tm.classify_retryable_upload_error(ERR_CURL_28), "network")
        self.assertEqual(tm.classify_retryable_upload_error(ERR_CURL_60), "network")

    def test_other_errors_are_not_retried(self):
        for message in (
            "上传失败: Bilibili登录态无效: 缺少必要 Cookie 字段: SESSDATA。请在设置页重新扫码登录后重试上传。",
            "上传失败: bilibili 账号不匹配：当前登录的 UID 是 111，设置要求的是 222，已阻止上传。",
            "上传失败: bilibili上传异常: 接口返回错误代码：21566，信息：标题含有敏感词",
            "上传失败: bilibili上传异常: bilibili上传被 preupload 接口返回 406 拒绝。",
            "",
            None,
        ):
            with self.subTest(message=message):
                self.assertIsNone(tm.classify_retryable_upload_error(message))


class UploadRetryPlanTests(unittest.TestCase):
    CONFIG = {"UPLOAD_AUTO_RETRY_ENABLED": True, "UPLOAD_AUTO_RETRY_MAX_ATTEMPTS": 3}
    NOW = datetime(2026, 11, 21, 19, 35, 54)

    def test_rate_limit_backoff_doubles(self):
        delays = []
        for count in range(3):
            plan = tm._plan_upload_auto_retry({"upload_retry_count": count}, ERR_137022, self.CONFIG, now=self.NOW)
            self.assertEqual(plan["attempt"], count + 1)
            self.assertEqual(plan["max_attempts"], 3)
            delays.append(plan["delay_minutes"])
        self.assertEqual(delays, [60, 120, 240])
        first = tm._plan_upload_auto_retry({"upload_retry_count": 0}, ERR_137022, self.CONFIG, now=self.NOW)
        self.assertEqual(first["retry_at"], "2026-11-21 20:35:54")

    def test_network_errors_retry_sooner(self):
        plan = tm._plan_upload_auto_retry({}, ERR_CURL_28, self.CONFIG, now=self.NOW)
        self.assertEqual(plan["kind"], "network")
        self.assertEqual(plan["delay_minutes"], 10)

    def test_limits_and_switch(self):
        self.assertIsNone(tm._plan_upload_auto_retry({"upload_retry_count": 3}, ERR_137022, self.CONFIG, now=self.NOW))
        self.assertIsNone(tm._plan_upload_auto_retry(
            {}, ERR_137022, {"UPLOAD_AUTO_RETRY_ENABLED": False}, now=self.NOW))
        self.assertIsNone(tm._plan_upload_auto_retry(
            {}, ERR_137022, {"UPLOAD_AUTO_RETRY_ENABLED": True, "UPLOAD_AUTO_RETRY_MAX_ATTEMPTS": 0}, now=self.NOW))
        self.assertIsNone(tm._plan_upload_auto_retry({}, "上传失败: 未知错误", self.CONFIG, now=self.NOW))

    def test_message_explains_the_retry(self):
        plan = tm._plan_upload_auto_retry({}, ERR_137022, self.CONFIG, now=self.NOW)
        message = tm._format_upload_retry_message(ERR_137022, plan)
        self.assertTrue(message.startswith(ERR_137022))
        self.assertIn("60 分钟后自动重试", message)
        self.assertIn("1/3", message)


class BilibiliPacingTests(unittest.TestCase):
    def tearDown(self):
        tm._set_last_bilibili_submit_at(None)

    def test_wait_seconds(self):
        self.assertEqual(tm._bilibili_upload_wait_seconds(None, 1000.0, 1800), 0)
        self.assertEqual(tm._bilibili_upload_wait_seconds(1000.0, 1600.0, 1800), 1200)
        self.assertEqual(tm._bilibili_upload_wait_seconds(1000.0, 3000.0, 1800), 0)
        self.assertEqual(tm._bilibili_upload_wait_seconds(1000.0, 1001.0, 0), 0)

    def test_wait_loop_reports_progress_until_the_slot_opens(self):
        clock = {"now": 10_000.0}
        tm._set_last_bilibili_submit_at(clock["now"] - 60)  # 1 分钟前刚投过稿

        def fake_wait(timeout):
            clock["now"] += timeout
            return False  # 未取消

        with patch.object(tm, "update_task") as update_task:
            ok = tm._wait_for_bilibili_upload_slot(
                "task-pacing", MagicMock(), {"BILIBILI_MIN_UPLOAD_INTERVAL_MINUTES": 3},
                clock=lambda: clock["now"], wait=fake_wait,
            )
        self.assertTrue(ok)
        self.assertGreaterEqual(clock["now"], 10_000.0 - 60 + 180)
        progress = [call.kwargs.get("upload_progress") for call in update_task.call_args_list]
        self.assertTrue(progress and all("投稿间隔" in text for text in progress), progress)

    def test_wait_loop_stops_when_cancelled(self):
        tm._set_last_bilibili_submit_at(10_000.0)
        with patch.object(tm, "update_task"):
            ok = tm._wait_for_bilibili_upload_slot(
                "task-pacing", MagicMock(), {"BILIBILI_MIN_UPLOAD_INTERVAL_MINUTES": 30},
                clock=lambda: 10_001.0, wait=lambda timeout: True,
            )
        self.assertFalse(ok)

    def test_zero_interval_never_waits(self):
        tm._set_last_bilibili_submit_at(10_000.0)
        wait = MagicMock(return_value=False)
        with patch.object(tm, "update_task") as update_task:
            ok = tm._wait_for_bilibili_upload_slot(
                "task-pacing", MagicMock(), {"BILIBILI_MIN_UPLOAD_INTERVAL_MINUTES": 0},
                clock=lambda: 10_001.0, wait=wait,
            )
        self.assertTrue(ok)
        wait.assert_not_called()
        update_task.assert_not_called()


class DefaultsTests(unittest.TestCase):
    """缺键的部分配置（测试替身、旧调用方）不触发等待和重试；load_config 补齐默认值后才默认开启。"""

    def tearDown(self):
        tm._set_last_bilibili_submit_at(None)

    def test_partial_config_neither_waits_nor_retries(self):
        tm._set_last_bilibili_submit_at(time.time())
        wait = MagicMock(return_value=False)
        with patch.object(tm, "update_task") as update_task:
            self.assertTrue(tm._wait_for_bilibili_upload_slot("task-defaults", MagicMock(), {}, wait=wait))
        wait.assert_not_called()
        update_task.assert_not_called()
        self.assertIsNone(tm._plan_upload_auto_retry({}, ERR_137022, {}))

    def test_shipped_defaults_turn_both_on(self):
        from modules.config_manager import DEFAULT_CONFIG
        self.assertEqual(DEFAULT_CONFIG["BILIBILI_MIN_UPLOAD_INTERVAL_MINUTES"], 30)
        self.assertTrue(DEFAULT_CONFIG["UPLOAD_AUTO_RETRY_ENABLED"])
        self.assertEqual(DEFAULT_CONFIG["UPLOAD_AUTO_RETRY_MAX_ATTEMPTS"], 3)
        self.assertTrue(DEFAULT_CONFIG["MONITOR_SKIP_UNTRANSLATABLE_TITLES"])
        self.assertEqual(DEFAULT_CONFIG["BILIBILI_EXPECTED_UID"], "")
        self.assertIsNotNone(tm._plan_upload_auto_retry({}, ERR_137022, DEFAULT_CONFIG))


class BilibiliAccountGuardTests(unittest.TestCase):
    def test_check_expected_account(self):
        credential = SimpleNamespace(dedeuserid="3546857477245382")
        self.assertEqual(check_expected_bilibili_account(credential, ""), (True, ""))
        self.assertEqual(check_expected_bilibili_account(credential, " 3546857477245382 "), (True, ""))
        ok, message = check_expected_bilibili_account(SimpleNamespace(dedeuserid="111"), "3546857477245382")
        self.assertFalse(ok)
        self.assertIn("111", message)
        self.assertIn("3546857477245382", message)
        ok, _ = check_expected_bilibili_account(object(), "3546857477245382")
        self.assertFalse(ok)

    def test_mismatched_account_blocks_upload_before_any_request(self):
        temp_dir = tempfile.mkdtemp(prefix="y2a-account-guard-")
        try:
            video_path = os.path.join(temp_dir, "video.mp4")
            cover_path = os.path.join(temp_dir, "cover.jpg")
            for path in (video_path, cover_path):
                with open(path, "wb") as stream:
                    stream.write(b"x")
            with patch("modules.bilibili_uploader.setup_task_logger", return_value=MagicMock()), \
                    patch("modules.bilibili_uploader.configure_bilibili_runtime"), \
                    patch("modules.bilibili_uploader.load_credential_from_file",
                          return_value=SimpleNamespace(dedeuserid="111")), \
                    patch("modules.bilibili_uploader.validate_credential_remote", return_value=(True, "ok")), \
                    patch("modules.bilibili_uploader.video_uploader.VideoUploader") as uploader_cls:
                ok, message = BilibiliUploader(
                    "cookies.json", expected_uid="3546857477245382",
                ).upload_video(video_path, cover_path, "标题", "简介", [], 21, task_id="test-guard")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        self.assertFalse(ok)
        self.assertIn("账号不匹配", message)
        uploader_cls.assert_not_called()
        self.assertIsNone(tm.classify_retryable_upload_error(f"上传失败: {message}"))


class _TempTaskDb(unittest.TestCase):
    """把 task_manager 指向一个临时 SQLite 库。"""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="y2a-hardening-db-")
        self._db_patch = patch.object(tm, "DB_PATH", os.path.join(self._tmp, "tasks.db"))
        self._db_patch.start()
        tm.init_db()

    def tearDown(self):
        self._db_patch.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _insert_task(self, task_id, status=None, **fields):
        columns = ["id", "youtube_url", "status"] + list(fields)
        values = [task_id, "https://www.youtube.com/watch?v=test", status or tm.TASK_STATES["FAILED"]]
        values += list(fields.values())
        conn = tm.get_db_connection()
        try:
            conn.execute(
                f"INSERT INTO tasks ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                values,
            )
            conn.commit()
        finally:
            conn.close()


class UploadRetryPersistenceTests(_TempTaskDb):
    def test_new_columns_exist_and_are_writable(self):
        self._insert_task("task-a")
        self.assertTrue(tm.update_task(
            "task-a", status=tm.TASK_STATES["FAILED"], upload_retry_count=1,
            upload_retry_at="2026-11-21 20:35:54", error_category=tm.UPLOAD_AUTO_RETRY_ERROR_CATEGORY))
        task = tm.get_task("task-a")
        self.assertEqual(task["upload_retry_count"], 1)
        self.assertEqual(task["upload_retry_at"], "2026-11-21 20:35:54")

    def test_any_other_status_change_cancels_a_pending_retry(self):
        self._insert_task("task-b")
        tm.update_task("task-b", status=tm.TASK_STATES["FAILED"], upload_retry_at="2026-11-21 20:35:54")
        tm.update_task("task-b", status=tm.TASK_STATES["READY_FOR_UPLOAD"])
        self.assertIsNone(tm.get_task("task-b")["upload_retry_at"])
        # 只改进度、不改状态时保留计划
        tm.update_task("task-b", status=tm.TASK_STATES["FAILED"], upload_retry_at="2026-11-21 21:00:00")
        tm.update_task("task-b", upload_progress=None, silent=True)
        self.assertEqual(tm.get_task("task-b")["upload_retry_at"], "2026-11-21 21:00:00")

    def test_due_retries_are_selected_by_time(self):
        self._insert_task("due", error_category=tm.UPLOAD_AUTO_RETRY_ERROR_CATEGORY,
                          upload_retry_at="2026-11-21 20:00:00")
        self._insert_task("later", error_category=tm.UPLOAD_AUTO_RETRY_ERROR_CATEGORY,
                          upload_retry_at="2026-11-21 23:00:00")
        self._insert_task("plain-failure", error_message="上传失败: 未知错误")
        due = tm._get_due_upload_retry_task_ids(now=datetime(2026, 11, 21, 21, 0, 0))
        self.assertEqual(due, ["due"])

    def test_last_successful_submit_is_read_back_from_the_db(self):
        self._insert_task("done", status=tm.TASK_STATES["COMPLETED"],
                          bilibili_upload_response='{"bvid": "BV1test"}',
                          updated_at="2026-11-21 19:33:43")
        loaded = tm._load_last_bilibili_submit_at_from_db()
        self.assertEqual(datetime.fromtimestamp(loaded), datetime(2026, 11, 21, 19, 33, 43))


if __name__ == "__main__":
    unittest.main()
