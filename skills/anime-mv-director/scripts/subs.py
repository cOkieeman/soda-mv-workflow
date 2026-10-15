"""第一夜 MV：字幕引擎。副歌直書（vert）、主歌依留白擺放（flow），日文都跟著歌聲逐字暈開浮現。

每個字先畫成小圖塊（清晰版＋墨暈版＋陰影），合成時依時間算每個字的透明度，畫在畫面上。
時間軸可用 sub_pos 覆蓋某句的位置：{起始秒數: {"pos": (x, y), "align": "left"|"center", "side": "right"|"left"}}
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).parent
SERIF = "C:/Windows/Fonts/NotoSerifSC-VF.ttf"
KAI = "C:/Windows/Fonts/simkai.ttf"  # 楷體：繁體字完整（瘦金體缺繁體字，不能用）
INK = (244, 238, 226)
W, H = 1920, 1080
PAD = 14


def _font(path, size, weight=None):
    f = ImageFont.truetype(path, size)
    if weight:
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
    return f


FJ_V, FZ_V = _font(SERIF, 52, 500), _font(KAI, 36)      # 直書
FJ_F, FZ_F = _font(SERIF, 58, 450), _font(KAI, 40)      # 依留白擺放


def _tile(ch, font, alpha=255):
    """一個字的圖塊：回傳 (清晰 RGBA, 墨暈 RGBA)，都已墊好柔和暗影，numpy float 0–1。"""
    w = int(font.getlength(ch)) + PAD * 2
    h = font.size + PAD * 2 + font.size // 4
    im = Image.new("RGBA", (max(w, 1), h), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((PAD, PAD), ch, font=font, fill=INK + (alpha,))
    a = im.getchannel("A")
    shadow = Image.new("RGBA", im.size, (6, 4, 2, 0))
    shadow.putalpha(a.filter(ImageFilter.GaussianBlur(6)).point(lambda v: min(255, int(v * 1.25))))
    sharp = Image.alpha_composite(shadow, im)
    soft = sharp.filter(ImageFilter.GaussianBlur(4))
    f = lambda x: np.asarray(x, np.float32) / 255.0
    return f(sharp), f(soft)


class Line:
    """一句字幕：預先排好每個字的位置與圖塊。"""

    def __init__(self, entry, override=None):
        o = override or {}
        self.t0, self.t1 = entry["t0"], entry["t1"]
        self.style = o.get("style", entry["style"])
        jp, zh = entry["jp"], entry["zh"]
        self.jp, self.zh = [], []   # [(x, y, sharp, soft)]
        if self.style == "vert":
            self._layout_vert(jp, zh, o)
        else:
            self._layout_flow(jp, zh, o)
        self._wash = None
        if self.style == "flow":  # 主歌字幕底下墊一層柔和暗影，亮背景也讀得清楚
            xs = [x for x, y, t, _ in self.jp + self.zh] + [x + t.shape[1] for x, y, t, _ in self.jp + self.zh]
            ys = [y for x, y, t, _ in self.jp + self.zh] + [y + t.shape[0] for x, y, t, _ in self.jp + self.zh]
            x0, x1, y0, y1 = max(min(xs) - 90, 0), min(max(xs) + 90, W), max(min(ys) - 50, 0), min(max(ys) + 50, H)
            u, v = np.meshgrid(np.linspace(-1, 1, x1 - x0), np.linspace(-1, 1, y1 - y0))
            m = np.clip(1 - np.maximum(np.abs(u) ** 6, np.abs(v) ** 2), 0, 1).astype(np.float32)
            self._wash = (x0, y0, m[..., None] * 0.42)
        dur = self.t1 - self.t0
        self.reveal = min(dur * 0.6, 0.26 * len(jp))   # 日文逐字浮現的總長
        self.zh_start = self.t0 + self.reveal * 0.5

    @staticmethod
    def _columns(text, per):
        """直書斷行：有逗號就在逗號後斷，否則每 per 個字一行；標點換成直排字形。"""
        vert = str.maketrans({"、": "︑", "，": "︐", "。": "︒", "　": " "})
        if len(text) > per:
            for p in ("、", "，", "　"):
                if p in text:
                    i = text.index(p) + 1
                    return [text[:i].translate(vert), text[i:].strip("　").translate(vert)]
        return [text[i:i + per].translate(vert) for i in range(0, len(text), per)]

    def _layout_vert(self, jp, zh, o):
        side = o.get("side", "right")
        cols = self._columns(jp, 15)              # 超過就換到左邊下一行
        x0 = o.get("pos", (1720 if side == "right" else 260, 150))[0]
        y0 = o.get("pos", (0, 150))[1]
        step = FJ_V.size * 1.1
        x = x0
        for col in cols:
            for i, ch in enumerate(col):
                s, so = _tile(ch, FJ_V)
                self.jp.append((int(x - s.shape[1] / 2), int(y0 + i * step), s, so))
            x -= FJ_V.size * 1.45
        zcols = [c.replace("︐", "") for c in self._columns(zh, 15)]  # 楷體沒有直排逗號，中文直書不加標點
        x = x - FJ_V.size * 0.1
        for col in zcols:
            for i, ch in enumerate(col):
                if ch == " ":
                    continue
                s, so = _tile(ch, FZ_V, 225)
                self.zh.append((int(x - s.shape[1] / 2), int(y0 + 40 + i * FZ_V.size * 1.2), s, so))
            x -= FZ_V.size * 1.5

    def _layout_flow(self, jp, zh, o):
        align = o.get("align", "center")
        x0, y0 = o.get("pos", (W / 2, 860))
        spacing = 8
        widths = [FJ_F.getlength(c) + spacing for c in jp]
        x = x0 - sum(widths) / 2 if align == "center" else x0
        for ch, w in zip(jp, widths):
            s, so = _tile(ch, FJ_F)
            self.jp.append((int(x - PAD), int(y0 - PAD), s, so))
            x += w
        zw = sum(FZ_F.getlength(c) + 3 for c in zh)
        x = x0 - zw / 2 if align == "center" else x0
        for ch in zh:
            s, so = _tile(ch, FZ_F, 225)
            self.zh.append((int(x - PAD), int(y0 + FJ_F.size * 1.45 - PAD), s, so))
            x += FZ_F.getlength(ch) + 3

    def draw(self, frame, t):
        """frame：BGR float 0–1（H×W×3），就地畫上。"""
        if not (self.t0 <= t < self.t1):
            return frame
        out_fade = min(1.0, (self.t1 - t) / 0.45)
        if self._wash is not None:
            x0, y0, m = self._wash
            k = min(1.0, (t - self.t0) / 0.6) * out_fade
            h, w = m.shape[:2]
            frame[y0:y0 + h, x0:x0 + w] *= (1 - m * k)
        n = len(self.jp)
        for i, (x, y, s, so) in enumerate(self.jp):
            ti = self.t0 + self.reveal * i / max(n, 1)
            a = min(1.0, max(0.0, (t - ti) / 0.45))
            self._blit(frame, x, y - int(8 * (1 - a)), s, so, a, out_fade)
        za = min(1.0, max(0.0, (t - self.zh_start) / 0.7))
        for x, y, s, so in self.zh:
            self._blit(frame, x, y, s, so, za, out_fade)
        return frame

    @staticmethod
    def _blit(frame, x, y, sharp, soft, a, fade):
        if a <= 0 or fade <= 0:
            return
        tile = sharp * a + soft * (1 - a)          # 剛浮現時帶墨暈，慢慢收成清晰的字
        h, w = tile.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x0 >= x1 or y0 >= y1:
            return
        t = tile[y0 - y:y1 - y, x0 - x:x1 - x]
        al = t[..., 3:4] * a * fade
        frame[y0:y1, x0:x1] = frame[y0:y1, x0:x1] * (1 - al) + t[..., [2, 1, 0]] * al


def lines_between(t0, t1, overrides=None):
    data = json.load(open(ROOT / "analysis" / "lyrics.json", encoding="utf-8"))
    ov = overrides or {}
    return [Line(e, ov.get(e["t0"])) for e in data if e["t1"] > t0 and e["t0"] < t1]


class Title(Line):
    """片名卡：大字楷體直書，一字一字慢慢浮現，旁邊一行小字。"""

    def __init__(self, t0, t1, text, sub="", pos=(360, 300)):
        self.t0, self.t1 = t0, t1
        self.jp, self.zh = [], []
        self._wash = None
        big = _font(KAI, 128)
        small = _font(SERIF, 30, 400)
        x, y = pos
        for i, ch in enumerate(text):
            s, so = _tile(ch, big)
            self.jp.append((int(x - s.shape[1] / 2), int(y + i * big.size * 1.12), s, so))
        sx = x - big.size * 1.0
        for i, ch in enumerate(sub):
            s, so = _tile(ch, small, 210)
            self.zh.append((int(sx - s.shape[1] / 2), int(y + 30 + i * small.size * 1.15), s, so))
        self.reveal = min((t1 - t0) * 0.5, 0.55 * len(text))
        self.zh_start = t0 + self.reveal


class Card:
    """黑底字卡：多行置中，整體淡入淡出。lines：[(文字, 字型種類 'jp'|'zh'|'small', 行距倍數)]"""

    def __init__(self, t0, t1, lines, y_center=540, fade=0.9):
        self.t0, self.t1, self.fade = t0, t1, fade
        fonts = {"jp": _font(SERIF, 44, 450), "zh": _font(KAI, 40), "big": _font(KAI, 64), "small": _font(SERIF, 26, 400)}
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        heights = [fonts[k].size * g for _, k, g in lines]
        y = y_center - sum(heights) / 2
        for (text, k, g), hgt in zip(lines, heights):
            if text:
                f = fonts[k]
                tw = d.textlength(text, font=f)
                d.text(((W - tw) / 2, y), text, font=f, fill=INK + (235 if k != "small" else 170,))
            y += hgt
        a = np.asarray(im, np.float32) / 255.0
        self.rgb, self.al = a[..., [2, 1, 0]], a[..., 3:4]

    def draw(self, frame, t):
        if not (self.t0 <= t < self.t1):
            return frame
        k = min(1.0, (t - self.t0) / self.fade, (self.t1 - t) / self.fade)
        frame[:] = frame * (1 - self.al * k) + self.rgb * self.al * k
        return frame
