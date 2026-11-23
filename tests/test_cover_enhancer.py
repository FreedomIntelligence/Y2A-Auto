"""投稿封面优化的测试：封面大字提炼、竖屏缩略图识别、版式输出、投稿流程里的开关。"""

import logging
import os
import random
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageFilter

from modules import cover_enhancer as ce
from modules.task_manager import TaskProcessor


def _noise(size, seed):
    rnd = random.Random(seed)
    img = Image.new("RGB", size)
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(size[0] * size[1])])
    return img


def _yellow_pixels(img, box):
    region = img.crop(box).convert("RGB")
    pixels = region.load()
    return sum(
        1
        for x in range(region.width)
        for y in range(region.height)
        if pixels[x, y][0] > 220 and pixels[x, y][1] > 190 and pixels[x, y][2] < 80
    )


class CoverTextTests(unittest.TestCase):
    def test_short_first_clause_is_used(self):
        self.assertEqual(ce.pick_cover_text("洗牙过程超解压"), "洗牙")
        self.assertEqual(ce.pick_cover_text("蛀牙是怎么形成的？从15000年前的头骨说起"), "蛀牙")

    def test_falls_back_to_most_specific_dental_term(self):
        self.assertEqual(ce.pick_cover_text("牙医揭秘根管治疗全过程"), "根管治疗")
        self.assertEqual(ce.pick_cover_text("种植牙是怎么放进去的"), "种植牙")
        self.assertEqual(ce.pick_cover_text("去除牙齿银汞合金充填物的过程"), "银汞合金")
        self.assertEqual(ce.pick_cover_text("深闪电状裂纹牙齿修复！⚡😱"), "裂纹牙")

    def test_no_text_when_nothing_fits(self):
        self.assertEqual(ce.pick_cover_text("上牙疼可能是鼻窦炎？上排牙痛与鼻窦炎的关系"), "")
        self.assertEqual(ce.pick_cover_text("Root canal"), "")
        self.assertEqual(ce.pick_cover_text(""), "")
        self.assertEqual(ce.pick_cover_text(None), "")


class VerticalStripTests(unittest.TestCase):
    def _pillarboxed(self, sides):
        width, height = 640, 360
        strip_w = round(height * 9 / 16)
        x0 = (width - strip_w) // 2
        canvas = sides.resize((width, height))
        canvas.paste(_noise((strip_w, height), 7), (x0, 0))
        return canvas, (x0, x0 + strip_w)

    def test_blurred_side_bars_are_detected(self):
        blurred = _noise((640, 360), 1).filter(ImageFilter.GaussianBlur(12))
        img, (x0, x1) = self._pillarboxed(blurred)
        strip = ce._vertical_strip(img)
        self.assertIsNotNone(strip)
        self.assertEqual(strip.size, (x1 - x0, 360))

    def test_black_side_bars_are_detected(self):
        img, _ = self._pillarboxed(Image.new("RGB", (640, 360), "black"))
        self.assertIsNotNone(ce._vertical_strip(img))

    def test_ordinary_landscape_frame_is_left_alone(self):
        self.assertIsNone(ce._vertical_strip(_noise((640, 360), 3)))
        smooth = Image.linear_gradient("L").resize((640, 360)).convert("RGB")
        self.assertIsNone(ce._vertical_strip(smooth))


class BuildCoverTests(unittest.TestCase):
    def test_vertical_source_puts_frame_left_and_text_right(self):
        cover = ce.build_cover(_noise((270, 480), 4), "根管治疗")
        self.assertEqual(cover.size, ce.COVER_SIZE)
        width, height = cover.size
        self.assertGreater(_yellow_pixels(cover, (width // 2, 0, width, height)), 500)
        self.assertEqual(_yellow_pixels(cover, (0, 0, int(width * 0.06), height)), 0)

    def test_vertical_source_without_text_is_centered(self):
        cover = ce.build_cover(Image.new("RGB", (270, 480), (200, 30, 30)), "")
        width, height = cover.size
        self.assertEqual(cover.getpixel((width // 2, height // 2)), (200, 30, 30))
        self.assertNotEqual(cover.getpixel((int(width * 0.1), height // 2)), (200, 30, 30))

    def test_landscape_source_fills_frame_with_text_bottom_left(self):
        source = Image.new("RGB", (1920, 1080), (40, 90, 160))
        cover = ce.build_cover(source, "洗牙")
        width, height = cover.size
        self.assertEqual(cover.size, ce.COVER_SIZE)
        self.assertEqual(cover.getpixel((width - 5, 5)), (40, 90, 160))
        self.assertGreater(_yellow_pixels(cover, (0, height // 2, width // 2, height)), 500)

    def test_enhance_cover_failure_returns_none(self):
        self.assertIsNone(ce.enhance_cover("/nonexistent/cover.jpg", "/nonexistent/out.jpg", "洗牙"))


class UploadHookTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cover = os.path.join(self.tmpdir, "cover.jpg")
        Image.new("RGB", (270, 480), (90, 90, 90)).save(self.cover)
        self.logger = logging.getLogger("cover-test")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _hook(self, enabled, cover_path):
        fake = SimpleNamespace(config={"BILIBILI_COVER_ENHANCE_ENABLED": enabled})
        return TaskProcessor._bilibili_upload_cover(fake, cover_path, "洗牙过程超解压", self.logger)

    def test_disabled_keeps_original(self):
        self.assertEqual(self._hook(False, self.cover), self.cover)
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "cover_bili.jpg")))

    def test_enabled_writes_a_separate_upload_cover(self):
        result = self._hook("on", self.cover)
        self.assertEqual(result, os.path.join(self.tmpdir, "cover_bili.jpg"))
        with Image.open(result) as img:
            self.assertEqual(img.size, ce.COVER_SIZE)
        with Image.open(self.cover) as original:
            self.assertEqual(original.size, (270, 480), "原封面文件不应被改动")

    def test_custom_cover_is_not_touched(self):
        custom = os.path.join(self.tmpdir, "custom_cover.png")
        Image.new("RGB", (640, 360)).save(custom)
        self.assertEqual(self._hook(True, custom), custom)

    def test_missing_cover_is_passed_through(self):
        self.assertEqual(self._hook(True, ""), "")
        missing = os.path.join(self.tmpdir, "nope.jpg")
        self.assertEqual(self._hook(True, missing), missing)


if __name__ == "__main__":
    unittest.main()
