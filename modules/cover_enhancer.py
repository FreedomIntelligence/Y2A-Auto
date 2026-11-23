"""投稿封面优化：把竖屏缩略图裁成铺满的 16:9 特写，并加上从标题提炼的中文大字。

B站封面按 16:9 显示。直接用竖屏 Shorts 的缩略图，画面会缩在中间一条，两边是空白或模糊边；
B站播放量高的牙科视频，封面多是铺满画面的特写加两三个醒目的中文大字。

只处理图片，不调用模型。失败时返回 None，调用方继续用原封面。
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageStat

logger = logging.getLogger(__name__)

COVER_SIZE: Tuple[int, int] = (1280, 720)
COVER_TEXT_MAX_CHARS = 6

_FONT_CANDIDATES = (
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fonts', 'SourceHanSansHWSC-VF.otf'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fonts', 'NotoSansCJKsc-Regular.otf'),
)

# 封面大字优先用的牙科术语：越具体越靠前；「牙齿」「口腔」这类泛词不当封面字
_COVER_TERMS = (
    '根管治疗', '种植牙', '隐形矫治', '牙齿矫正', '牙周病', '牙结石', '牙菌斑', '智齿', '洗牙', '补牙',
    '拔牙', '银汞合金', '牙冠', '瓷贴面', '贴面', '嵌体', '义齿', '假牙', '蛀牙', '龋齿', '牙套', '正畸', '牙龈',
    '牙髓', '脓肿', '牙齿美白', '美白', '裂纹牙', '根尖',
)
_SPLIT_RE = re.compile(r'[：:，,。！!？?｜|—\-（(）)\[\]【】「」“”"\s·/]+')
_CJK_RE = re.compile(r'[一-鿿]')
_EMOJI_RE = re.compile('[\U0001F000-\U0001FFFF☀-➿️‍]')


def pick_cover_text(title: str, max_chars: int = COVER_TEXT_MAX_CHARS) -> str:
    """从投稿标题提炼封面大字：标题第一个短句够短就用它，否则用标题里最具体的牙科术语，都没有就不加字。"""
    text = _EMOJI_RE.sub('', str(title or '')).strip()
    if not text:
        return ''
    first = _SPLIT_RE.split(text, maxsplit=1)[0].strip()
    if 2 <= len(first) <= max_chars and _CJK_RE.search(first):
        return first
    for term in _COVER_TERMS:
        if term in text:
            return term
    return ''


def _column_means(gray: Image.Image, x0: int, x1: int) -> float:
    if x1 <= x0:
        return 0.0
    return ImageStat.Stat(gray.crop((x0, 0, x1, gray.height))).mean[0]


def _trim_dark_side_bars(img: Image.Image) -> Image.Image:
    """去掉左右两侧的黑边。"""
    gray = img.convert('L')
    width, height = gray.size
    content = [x for x in range(width) if _column_means(gray, x, x + 1) > 18]
    if not content:
        return img
    left, right = content[0], content[-1] + 1
    if right - left < width * 0.85:
        return img.crop((left, 0, right, height))
    return img


def _vertical_strip(img: Image.Image) -> Optional[Image.Image]:
    """横版缩略图里是不是「中间一条竖屏画面、两边模糊或黑边」：是就返回中间那条。

    YouTube 给 Shorts 生成的 16:9 缩略图常是这样。判断依据：中间 9:16 区域比两侧清晰得多，
    且两条分界线上有明显的竖直接缝（自然拍摄的虚化背景没有这种笔直的接缝）。
    """
    width, height = img.size
    if width / height < 1.5:
        return None
    strip_w = round(height * 9 / 16)
    x0 = (width - strip_w) // 2
    x1 = x0 + strip_w
    if x0 < 16:
        return None
    edges = img.convert('L').filter(ImageFilter.FIND_EDGES)
    center = _column_means(edges, x0 + 6, x1 - 6)
    sides = (_column_means(edges, 0, x0 - 6) + _column_means(edges, x1 + 6, width)) / 2
    seams = []
    for seam_x, outside in ((x0, (x0 - 14, x0 - 4)), (x1, (x1 + 4, x1 + 14))):
        seam = max(_column_means(edges, x, x + 1) for x in range(seam_x - 2, seam_x + 2))
        seams.append(seam > 3 * max(_column_means(edges, *outside), 1.0))
    if center > 2.5 * max(sides, 0.5) and all(seams):
        return img.crop((x0, 0, x1, height))
    return None


def _crop_to_ratio(img: Image.Image, ratio: float, center_y: float = 0.5) -> Image.Image:
    width, height = img.size
    if width / height < ratio:
        crop_h = max(1, round(width / ratio))
        top = int(min(max(height * center_y - crop_h / 2, 0), height - crop_h))
        return img.crop((0, top, width, top + crop_h))
    crop_w = max(1, round(height * ratio))
    left = (width - crop_w) // 2
    return img.crop((left, 0, left + crop_w, height))


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if not os.path.isfile(path):
            continue
        font = ImageFont.truetype(path, size)
        try:
            font.set_variation_by_name('Bold')
        except Exception:
            pass
        return font
    return ImageFont.load_default()


def _split_lines(text: str) -> List[str]:
    if len(text) <= 4:
        return [text]
    half = (len(text) + 1) // 2
    return [text[:half], text[half:]]


def _draw_text_block(img: Image.Image, lines: List[str], box: Tuple[int, int, int, int],
                     max_size: int, align: str) -> None:
    """在 box 里画黄字黑描边，字号取放得下的最大值；align 为 'center' 或 'bottom-left'。"""
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = box
    size = max_size
    while True:
        font = _load_font(size)
        stroke = max(4, size // 9)
        spacing = int(size * 0.12)
        bbox = draw.multiline_textbbox((0, 0), '\n'.join(lines), font=font, stroke_width=stroke,
                                       spacing=spacing, align='center')
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if (text_w <= right - left and text_h <= bottom - top) or size <= 28:
            break
        size = int(size * 0.92)
    if align == 'center':
        x = left + (right - left - text_w) // 2 - bbox[0]
        y = top + (bottom - top - text_h) // 2 - bbox[1]
    else:
        x = left - bbox[0]
        y = bottom - text_h - bbox[1]
    draw.multiline_text((x, y), '\n'.join(lines), font=font, fill=(255, 226, 0), stroke_width=stroke,
                        stroke_fill=(0, 0, 0), spacing=spacing, align='center')


def _compose_vertical(frame: Image.Image, text: str, size: Tuple[int, int]) -> Image.Image:
    """竖屏画面：虚化压暗的同图做背景，画面按 3:4 轻裁后放左侧，大字放右侧空白处。"""
    width, height = size
    background = _crop_to_ratio(frame, width / height).resize(size, Image.LANCZOS)
    background = ImageEnhance.Brightness(background.filter(ImageFilter.GaussianBlur(24))).enhance(0.45)
    foreground = _crop_to_ratio(frame, 3 / 4) if frame.width / frame.height < 3 / 4 else frame
    fg_h = int(height * 0.92)
    fg_w = max(1, round(foreground.width * fg_h / foreground.height))
    foreground = foreground.resize((fg_w, fg_h), Image.LANCZOS)
    if text:
        x = int(width * 0.06)
    else:
        x = (width - fg_w) // 2
    y = (height - fg_h) // 2
    shadow = Image.new('L', size, 0)
    ImageDraw.Draw(shadow).rectangle((x + 8, y + 10, x + fg_w + 8, y + fg_h + 10), fill=160)
    background.paste((0, 0, 0), (0, 0), shadow.filter(ImageFilter.GaussianBlur(12)))
    background.paste(foreground, (x, y))
    if text:
        text_left = x + fg_w + int(width * 0.04)
        _draw_text_block(background, _split_lines(text), (text_left, int(height * 0.12), int(width * 0.96),
                         int(height * 0.88)), int(height * 0.3), 'center')
    return background


def build_cover(img: Image.Image, text: str, size: Tuple[int, int] = COVER_SIZE) -> Image.Image:
    img = img.convert('RGB')
    vertical = _vertical_strip(img)
    if vertical is None:
        img = _trim_dark_side_bars(img)
        if img.width / img.height < 1.2:
            vertical = img
    if vertical is not None:
        return _compose_vertical(vertical, text, size)
    cover = _crop_to_ratio(img, size[0] / size[1]).resize(size, Image.LANCZOS)
    if text:
        width, height = size
        # 底部渐暗，压住原片底部的外文字幕或标题，让黄字更醒目
        shade = Image.new('L', (1, height), 0)
        for y in range(height):
            shade.putpixel((0, y), int(170 * max(0.0, (y / height - 0.5) / 0.5) ** 1.2))
        cover.paste((0, 0, 0), (0, 0), shade.resize(size))
        _draw_text_block(cover, [text], (int(width * 0.045), int(height * 0.55), int(width * 0.8),
                         int(height * 0.93)), int(height * 0.2), 'bottom-left')
    return cover


def enhance_cover(src_path: str, out_path: str, title: str) -> Optional[str]:
    """生成优化后的封面，成功返回 out_path，失败返回 None（调用方继续用原封面）。"""
    try:
        with Image.open(src_path) as source:
            cover = build_cover(source, pick_cover_text(title))
        cover.save(out_path, 'JPEG', quality=90)
        return out_path
    except Exception as exc:
        logger.warning(f"封面优化失败，继续用原封面: {exc}")
        return None
