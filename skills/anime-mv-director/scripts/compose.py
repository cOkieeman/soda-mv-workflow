"""第一夜 MV：2.5D 合成引擎。依時間軸逐格合成 1920×1080，對上音軌與雙語字幕，交給 ffmpeg 編碼。

  python compose.py s1            # 輸出 out/s1.mp4（時間軸定義在 timelines.py）
  python compose.py s1 --preview  # 960×540、12fps 快速預覽

鏡頭（shot）欄位：
  t0, t1        起訖秒數（歌曲時間）
  src           單張圖路徑；或 seq＝逐格資料夾（自動重新取樣成鏡頭長度）
  xfade         [(秒, 圖路徑), ...] 鏡頭內在指定時間交叉淡化到下一張（表情差分、影子出現）
  xdur          交叉淡化長度（秒，預設 1.2）
  cam           (z0, z1, (cx0, cy0), (cx1, cy1))：縮放與畫面中心（0–1）從頭到尾緩動
  depth         深度圖路徑（越亮越近）＋ par（視差強度）→ 推鏡時近景放大得比遠景多
  fx            [("flicker", x, y, r, (R,G,B), 強度), ("sweep", 方向, (R,G,B), 強度, t0, t1),
                 ("dust", 數量, (R,G,B)), ("sparkle", x, y, t)]
  tin           進場轉場：("cut",) ("dissolve", 秒) ("flash", 秒, (R,G,B)) ("dip", 秒)
"""
import argparse, json, math, subprocess, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import imageio_ffmpeg

ROOT = Path(__file__).parent
W, H = 1920, 1080
FONT = "C:/Windows/Fonts/NotoSerifSC-VF.ttf"
_cache = {}


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def hd(path):
    """有 SeedVR2 去噪放大版（renders/hd/<名>_seedvr2.png）就改用它。"""
    p = ROOT / "renders" / "hd" / f"{Path(path).stem}_seedvr2.png"
    return str(p.relative_to(ROOT)) if p.exists() and "frame_" not in path else path


def load(path):
    path = hd(path)
    if path not in _cache:
        im = cv2.imdecode(np.fromfile(str(ROOT / path), np.uint8), cv2.IMREAD_COLOR)  # 路徑有中文，不能用 imread
        _cache[path] = im.astype(np.float32) / 255.0
    return _cache[path]


def load_depth(path, shape):
    key = (path, shape)
    if key not in _cache:
        d = cv2.imdecode(np.fromfile(str(ROOT / path), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        _cache[key] = cv2.GaussianBlur(cv2.resize(d, (shape[1], shape[0])), (0, 0), 6)
    return _cache[key]


def seq_frames(folder):
    if folder not in _cache:
        _cache[folder] = sorted((ROOT / folder).glob("frame_*.png"))
    return _cache[folder]


def camera(img, z, c, depth=None, par=0.0, z0=1.0, rot=0.0):
    """把 img 依縮放 z、中心 c 取景成 W×H；有深度圖時近景多放大（推鏡視差）；rot＝畫面旋轉角度（度）。"""
    h, w = img.shape[:2]
    s = max(W / w, H / h) * z
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    cx, cy = c[0] * w, c[1] * h
    # 中心要讓畫面不出界
    cx = min(max(cx, W / s / 2), w - W / s / 2)
    cy = min(max(cy, H / s / 2), h - H / s / 2)
    du, dv = u - W / 2, v - H / 2
    if rot:
        a = math.radians(rot)
        du, dv = du * math.cos(a) - dv * math.sin(a), du * math.sin(a) + dv * math.cos(a)
    mx = cx + du / s
    my = cy + dv / s
    if depth is not None and par:
        d = cv2.remap(depth, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        k = 1.0 + par * (z - z0) * (d - 0.5)
        mx = cx + (mx - cx) / k
        my = cy + (my - cy) / k
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # 來源圖座標（0–1）→ 輸出畫面座標（0–1），讓特效跟著鏡頭走
    to_out = lambda x, y: (((x * w - cx) * s + W / 2) / W, ((y * h - cy) * s + H / 2) / H)
    return out, to_out, s


_flow = {}


def flow_interp(a, b, f):
    """光流補中間格：a→b 之間 f（0–1）的位置。比直接疊化少鬼影，讓 24fps 片段可以輸出 60fps。"""
    key = (id(a), id(b))
    if key not in _flow:
        _flow.clear()
        ga = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        _flow[key] = (dis.calc(ga, gb, None), dis.calc(gb, ga, None))
    fab, fba = _flow[key]
    h, w = a.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    wa = cv2.remap(a, gx - fab[..., 0] * f, gy - fab[..., 1] * f, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wb = cv2.remap(b, gx - fba[..., 0] * (1 - f), gy - fba[..., 1] * (1 - f), cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    return wa * (1 - f) + wb * f


def base_image(shot, t):
    """鏡頭在時間 t 的原始畫面（未取景）。"""
    if "seq" in shot:
        fs = seq_frames(shot["seq"])
        if shot.get("frames"):  # 只取一段（例如動作還沒漂移的前段）
            fs = fs[shot["frames"][0] - 1:shot["frames"][1]]
        q = min(max((t - shot["t0"]) / (shot["t1"] - shot["t0"]), 0.0), 1.0)
        if shot.get("pingpong"):  # 正放再倒放，最後一格回到第一格的姿勢，才能和下一鏡無縫接
            q = 1 - abs(1 - 2 * q)
        x = q * (len(fs) - 1)
        i = int(x)
        a = load(str(fs[i].relative_to(ROOT)))
        if i + 1 < len(fs) and x - i > 0.02:
            b = load(str(fs[i + 1].relative_to(ROOT)))
            a = flow_interp(a, b, x - i)
        if shot.get("resize"):  # 圖轉影片的比例跟原圖不同時，拉回原圖比例才能和下一鏡對齊
            a = cv2.resize(a, tuple(shot["resize"]), interpolation=cv2.INTER_CUBIC)
        if len(_cache) > 400:  # 逐格素材不要一直佔記憶體
            for k in [k for k in _cache if isinstance(k, str) and "/frame_" in k.replace("\\", "/")][:200]:
                _cache.pop(k, None)
        return a, shot["seq"]
    imgs = [(shot["t0"], shot["src"])] + list(shot.get("xfade", []))
    cur, nxt, tt = imgs[0][1], None, 0.0
    for (ta, pa), (tb, pb) in zip(imgs, imgs[1:]):
        if t >= tb:
            cur = pb
    xd = shot.get("xdur", 1.2)
    for tb, pb in imgs[1:]:
        if tb - xd / 2 <= t < tb + xd / 2:
            prev = [p for ta, p in imgs if ta < tb][-1]
            a, b = load(prev), load(pb)
            if a.shape != b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]))
            f = ease((t - (tb - xd / 2)) / xd)
            return a * (1 - f) + b * f, prev
    return load(cur), cur


def render_shot(shot, t, rng):
    img, key = base_image(shot, t)
    p = (t - shot["t0"]) / (shot["t1"] - shot["t0"])
    p = min(max(p, -0.3), 1.3)  # 轉場重疊時繼續沿同方向運動
    if shot.get("ease_in"):     # 段落第一鏡才緩起
        p = p * p * (2 - p) if 0 <= p <= 1 else p
    z0, z1, c0, c1 = shot.get("cam", (1.0, 1.08, (0.5, 0.5), (0.5, 0.5)))
    z = max(1.0, z0 + (z1 - z0) * p)
    c = (c0[0] + (c1[0] - c0[0]) * p, c0[1] + (c1[1] - c0[1]) * p)
    if shot.get("black"):
        return np.zeros((H, W, 3), np.float32)
    depth = load_depth(shot["depth"], img.shape[:2]) if shot.get("depth") else None
    r0, r1 = shot.get("rot", (0.0, 0.0))
    out, to_out, scale = camera(img, z, c, depth, shot.get("par", 0.0), z0, r0 + (r1 - r0) * p)
    for fx in shot.get("fx", []):
        out = apply_fx(out, fx, t, shot, rng, to_out, scale)
    return out


_noise = None
_dust_pts = None
_petal_sprites = None


def petal_sprites():
    """程式繪製的花瓣（淚滴形、帶漸層與中脈），紫、白各一。回傳 BGR＋alpha float。"""
    global _petal_sprites
    if _petal_sprites is None:
        _petal_sprites = []
        S = 160
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S
        u, v = (xx - 0.5) * 2, yy                      # v：0＝花瓣尖、1＝花瓣圓端
        width = 0.62 * np.sin(np.clip(v, 0, 1) * np.pi * 0.92) ** 0.8
        inside = np.clip((width - np.abs(u)) * 14, 0, 1) * (v > 0.02) * (v < 0.98)
        alpha = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 1.5)
        shade = 0.82 + 0.18 * v - 0.12 * np.abs(u) - 0.10 * np.exp(-(u / 0.05) ** 2) * (v > 0.1)
        for base in ((150, 110, 205), (245, 240, 236)):   # 紫、白（RGB）
            col = np.dstack([np.full((S, S), c / 255.0, np.float32) for c in base[::-1]]) * shade[..., None]
            edge = np.clip(1 - (width - np.abs(u)) * 6, 0, 1)[..., None] * 0.15
            col = np.clip(col + edge, 0, 1)
            _petal_sprites.append(cv2.GaussianBlur(np.dstack([col, alpha * 0.82]), (0, 0), 1.6))
    return _petal_sprites


def draw_petals(out, t, t0, n, mode, seed=3, size=(34, 70), dur=None, cx=0.5, cy=0.5):
    """花瓣粒子。mode：fall＝緩緩飄落；burst＝從 (cx,cy) 散出後飄落；swirl＝由下往上捲起成旋風。
    只有 15 片紫白花瓣是故事設定，n 預設用 15。"""
    tt = t - t0
    if tt < 0:
        return out
    sprites = petal_sprites()
    rng = np.random.default_rng(seed)
    for i in range(n):
        sp = sprites[i % 2]
        sz = rng.uniform(*size)
        ph, spin, sway = rng.uniform(0, 6.28), rng.uniform(-2.2, 2.2), rng.uniform(0.02, 0.06)
        delay = rng.uniform(0, 1.2)
        u = tt - delay
        if u < 0:
            continue
        if mode == "fall":
            x = rng.uniform(-0.05, 1.05) + sway * math.sin(u * 1.3 + ph) + 0.015 * u
            y = -0.1 + rng.uniform(0, 0.4) + u * rng.uniform(0.05, 0.11)
        elif mode == "burst":
            ang, v = rng.uniform(0, 6.28), rng.uniform(0.08, 0.22)
            k = 1 - math.exp(-u * 1.6)
            x = cx + math.cos(ang) * v * k * 1.6 + sway * math.sin(u + ph)
            y = cy + math.sin(ang) * v * k + 0.05 * u * u
        else:  # swirl
            r = 0.08 + 0.35 * rng.random() + 0.04 * u
            ang = ph + u * (1.4 + rng.random())
            x = cx + r * math.cos(ang) * 0.9
            y = 1.1 - u * rng.uniform(0.12, 0.22) + 0.06 * math.sin(ang)
        if not (-0.1 < x < 1.1 and -0.15 < y < 1.15):
            continue
        h0, w0 = sp.shape[:2]
        scale = sz / max(h0, w0)
        M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), math.degrees(ph + spin * u), scale)
        M[0, 2] += x * W - w0 / 2
        M[1, 2] += y * H - h0 / 2
        tile = cv2.warpAffine(sp, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        a = tile[..., 3:4] * 0.95
        out = out * (1 - a) + tile[..., :3] * a
    return out


def dust_layer(t, n, col):
    """全片共用的浮塵：用歌曲時間計算，切鏡時不中斷。"""
    global _dust_pts
    if _dust_pts is None or len(_dust_pts) != n:
        _dust_pts = np.random.default_rng(11).random((n, 5))
    layer = np.zeros((H, W), np.float32)
    for px, py, sp, ph, sz in _dust_pts:
        x = int(((px + 0.010 * t * (0.4 + sp) + 0.02 * math.sin(t * 0.3 + ph * 6)) % 1) * W)
        y = int(((py - 0.016 * t * (0.3 + sp)) % 1) * H)
        a = (0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * (0.8 + 1.6 * sp) + ph * 6.28))) * (0.4 + 0.6 * sz)
        cv2.circle(layer, (x, y), 1 + int(sz * 2.4), a, -1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 2.0)
    return layer[..., None] * np.array(col[::-1], np.float32) / 255.0


def apply_fx(out, fx, t, shot, rng, to_out=lambda x, y: (x, y), scale=1.0):
    """flicker／sparkle 的 x、y 是來源圖座標（0–1），會跟著鏡頭換算。"""
    kind = fx[0]
    u, v = np.meshgrid(np.linspace(0, 1, W, dtype=np.float32), np.linspace(0, 1, H, dtype=np.float32))
    if kind == "flicker":  # 燈火呼吸：以 (x,y) 為中心的暖光，強度隨低頻雜訊抖動
        _, x, y, r, col, k = fx
        x, y = to_out(x, y)
        n = 0.75 + 0.25 * math.sin(t * 7.3) * math.sin(t * 2.9 + 1) + 0.08 * math.sin(t * 23.0)
        g = np.exp(-(((u - x) * 16 / 9) ** 2 + (v - y) ** 2) / (r * r))[..., None]
        return out + g * np.array(col[::-1], np.float32) / 255.0 * k * n
    if kind == "sweep":  # 燈光從一側掃入（有人提燈走近）
        _, side, col, k, ta, tb = fx
        f = ease((t - ta) / (tb - ta))
        ramp = (u if side == "right" else 1 - u) ** 1.6
        return out * (1 + ramp[..., None] * f * k * 0.6) + ramp[..., None] * f * k * np.array(col[::-1], np.float32) / 255.0 * 0.25
    if kind == "dust":  # 浮塵／螢光，固定種子的粒子緩慢漂移
        _, nmax, col = fx
        prng = np.random.default_rng(len(shot.get("src", shot.get("seq", ""))))
        pts = prng.random((nmax, 4))
        layer = np.zeros((H, W), np.float32)
        tt = t - shot["t0"]
        for px, py, sp, ph in pts:
            x = int(((px + 0.012 * tt * (0.5 + sp)) % 1) * W)
            y = int(((py - 0.02 * tt * (0.3 + sp)) % 1) * H)
            a = 0.5 + 0.5 * math.sin(tt * (1 + 2 * sp) + ph * 6.28)
            cv2.circle(layer, (x, y), 1 + int(sp * 2.5), a, -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 2.2)[..., None]
        return out + layer * np.array(col[::-1], np.float32) / 255.0 * 0.9
    if kind == "rain":  # 雨絲：兩層斜線（近大遠小），用歌曲時間計算，切鏡不中斷
        _, n, k = fx
        layer = np.zeros((H, W), np.float32)
        prng = np.random.default_rng(5)
        pts = prng.random((n, 3))
        for px, py, z in pts:
            spd, ln = 900 + 1500 * z, 18 + 50 * z
            x = (px * (W + 300) + t * spd * 0.18) % (W + 300) - 150
            y = (py * (H + 200) + t * spd) % (H + 200) - 100
            cv2.line(layer, (int(x), int(y)), (int(x - ln * 0.18), int(y - ln)), 0.25 + 0.6 * z, 1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 0.8)[..., None]
        return out * (1 - layer * k * 0.25) + layer * k * np.array([0.92, 0.9, 0.85], np.float32)
    if kind == "petals":  # ("petals", t0, n, mode, cx, cy)
        _, t0, n, mode, cx, cy = fx
        return draw_petals(out, t, t0, n, mode, seed=int(t0 * 10) % 97, cx=cx, cy=cy)
    if kind == "dream":  # 夢境：亮處大片暈光、輕微柔焦、四周淡出成霧，k＝強度 0–1
        _, k = fx
        hi = np.clip(out - 0.3, 0, None)
        glow = cv2.GaussianBlur(hi, (0, 0), 14) * 1.3 + cv2.GaussianBlur(hi, (0, 0), 48) * 1.8
        soft = cv2.GaussianBlur(out, (0, 0), 2.2)
        o = out * (1 - 0.35 * k) + soft * 0.35 * k                  # 輕微柔焦
        o = 1 - (1 - o) * (1 - np.clip(glow, 0, 1) * k)             # screen：光暈
        o = o * (1 - 0.06 * k) + 0.06 * k                            # 極輕抬暗部
        r = np.sqrt((u - 0.5) ** 2 * 1.3 + (v - 0.5) ** 2)
        edge = (np.clip((r - 0.38) / 0.32, 0, 1) ** 1.5)[..., None] * 0.6 * k
        mist = cv2.GaussianBlur(o, (0, 0), 30) * 0.6 + 0.4 * np.array([0.95, 0.93, 0.92], np.float32)
        return o * (1 - edge) + mist * edge
    if kind == "sparkle":  # 眼淚反光：一閃而過的星芒
        _, x, y, ts = fx
        x, y = to_out(x, y)
        a = math.exp(-((t - ts) / 0.18) ** 2)
        if a < 0.01:
            return out
        star = np.zeros((H, W), np.float32)
        cx, cy = int(x * W), int(y * H)
        cv2.line(star, (cx - 26, cy), (cx + 26, cy), 1.0, 1, cv2.LINE_AA)
        cv2.line(star, (cx, cy - 16), (cx, cy + 16), 1.0, 1, cv2.LINE_AA)
        star = cv2.GaussianBlur(star, (0, 0), 2) * 1.4 + cv2.GaussianBlur(star, (0, 0), 9) * 1.6
        return out + (star * a)[..., None]
    return out


GRAIN = 0.006  # 顆粒強度（使用者嫌噪點多，從 0.018 調低）；timeline 可用 "grain" 覆蓋


def grade(img, rng, grain=None):
    """全片共用調色：往原版 CG 的低飽和、深陰影靠，加暗角、顆粒。"""
    global _noise
    gray = img.mean(axis=2, keepdims=True)
    img = gray + (img - gray) * 0.9                       # 微降飽和
    img = np.clip(img, 0, None) ** 1.06                   # 壓一點暗部
    u, v = np.meshgrid(np.linspace(-1, 1, W, dtype=np.float32), np.linspace(-1, 1, H, dtype=np.float32))
    img = img * (1 - 0.28 * np.clip(u * u * 0.6 + v * v * 0.9 - 0.15, 0, 1))[..., None]
    if _noise is None:
        _noise = [rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]
    n = cv2.resize(_noise[rng.integers(6)], (W, H))[..., None]
    return np.clip(img + n * (GRAIN if grain is None else grain), 0, 1)


def sub_image(jp, zh):
    """一句雙語字幕的 RGBA 圖（日文大、中文小，帶柔和陰影）。"""
    fj = ImageFont.truetype(FONT, 50); fj.set_variation_by_axes([500])
    fz = ImageFont.truetype(FONT, 30); fz.set_variation_by_axes([400])
    im = Image.new("RGBA", (W, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for text, font, y in ((jp, fj, 40), (zh, fz, 118)):
        tw = d.textlength(text, font=font)
        d.text(((W - tw) / 2, y), text, font=font, fill=(244, 238, 226, 255))
    shadow = Image.new("RGBA", im.size, (0, 0, 0, 0))
    shadow.putalpha(im.getchannel("A").filter(ImageFilter.GaussianBlur(6)).point(lambda a: int(a * 0.85)))
    base = Image.new("RGBA", im.size, (0, 0, 0, 0))
    base = Image.alpha_composite(base, Image.new("RGBA", im.size, (8, 6, 4, 0)))
    base.putalpha(shadow.getchannel("A"))
    out = Image.alpha_composite(base, im)
    a = np.array(out).astype(np.float32) / 255.0
    return a[..., [2, 1, 0]], a[..., 3:4]


def luminance(img):
    return cv2.GaussianBlur(img @ np.array([0.114, 0.587, 0.299], np.float32), (0, 0), 9)


def blend(prev, cur, f, tin):
    """f：0→1 的轉場進度。prev、cur 都是各自鏡頭在同一時間點的畫面（都在動）。"""
    kind = tin[0]
    if kind in ("luma", "luma_inv"):  # luma：暗部先溶（像墨暈開）；luma_inv：亮部先溶，暗部留到最後
        soft = 0.35
        L = luminance(prev)
        if kind == "luma_inv":
            L = 1 - L
        m = np.clip((f * (1 + soft) - L) / soft, 0, 1)[..., None]
        return prev * (1 - m) + cur * m
    if kind == "bloom":     # 光源的光暈擴散，從光裡浮出下一鏡
        _, d, x, y, col = tin
        u, v = np.meshgrid(np.linspace(0, 1, W, dtype=np.float32), np.linspace(0, 1, H, dtype=np.float32))
        dist = np.sqrt(((u - x) * 16 / 9) ** 2 + (v - y) ** 2)
        r = ease(f) * 2.4
        m = np.clip((r - dist) / 0.6, 0, 1)[..., None]
        glow = np.exp(-(dist / (0.15 + r * 0.5)) ** 2)[..., None] * math.sin(math.pi * f) * 0.55
        c = np.array(col[::-1], np.float32) / 255.0
        return prev * (1 - m) + cur * m + glow * c
    if kind == "dim":       # 一邊溶一邊微微暗下去，像回憶淡出再浮現
        g = ease(f)
        return (prev * (1 - g) + cur * g) * (1 - 0.5 * math.sin(math.pi * f))
    f = ease(f)             # dissolve
    return prev * (1 - f) + cur * f


def apply_transition(frame_in, frame_prev, shot, t):
    tin = shot.get("tin", ("cut",))
    dt = t - shot["t0"]
    if tin[0] == "dissolve" and frame_prev is not None and dt < tin[1]:
        f = ease(dt / tin[1])
        return frame_prev * (1 - f) + frame_in * f
    if tin[0] == "flash":
        f = math.exp(-((dt) / (tin[1] * 0.4)) ** 2) * 0.42  # 峰值壓低，只是一道暖光閃過
        return frame_in * (1 - f * 0.4) + np.array(tin[2][::-1], np.float32) / 255.0 * f
    if tin[0] == "dip":
        f = min(1.0, dt / tin[1])
        return frame_in * ease(f)
    return frame_in


def main():
    from timelines import TIMELINES
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--test", type=float, default=0, help="只每 N 秒抽一格跑一遍，檢查會不會出錯，不輸出影片")  # 全片完成後再評估 60fps
    a = ap.parse_args()
    tl = TIMELINES[a.name]
    fps = a.fps
    shots, t_start, t_end = tl["shots"], tl["t0"], tl["t1"]
    import subs as subs_mod  # 字幕從 analysis/lyrics.json 自動取這段的歌詞；sub_pos 可覆蓋位置與樣式
    sub_lines = subs_mod.lines_between(t_start, t_end, tl.get("sub_pos")) if tl.get("audio") else []
    sub_lines += [subs_mod.Title(*ti) for ti in tl.get("titles", [])]
    sub_lines += [subs_mod.Card(*c) for c in tl.get("cards", [])]
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    path = out / f"{a.name}{'_preview' if a.preview else ''}.mp4"
    ow, oh = (960, 540) if a.preview else (W, H)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", str(fps),
           "-i", "-"] + (["-ss", str(t_start), "-t", str(t_end - t_start), "-i", str(ROOT / tl["audio"])] if tl.get("audio")
           else ["-f", "lavfi", "-t", str(t_end - t_start), "-i", "anullsrc=r=48000:cl=stereo"]) + [
           "-af", f"afade=t=in:d=0.4,afade=t=out:st={t_end - t_start - 0.8}:d=0.8",
           "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
           "-shortest", str(path)]
    if a.test:
        class _Null:
            def write(self, b): pass
            def close(self): pass
        proc = type("P", (), {"stdin": _Null(), "wait": lambda self: 0})()
        fps = 1 / a.test
    else:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rng = np.random.default_rng(7)
    n = int(round((t_end - t_start) * fps))
    for i in range(n):
        t = t_start + i / fps
        frame = None
        for k in range(1, len(shots)):  # 轉場窗以剪接點為中心，前後兩鏡同時在動
            tin = shots[k].get("tin", ("cut",))
            if tin[0] in ("dissolve", "luma", "luma_inv", "bloom", "dim"):
                d = tin[1]
                if shots[k]["t0"] - d / 2 <= t < shots[k]["t0"] + d / 2:
                    f = (t - (shots[k]["t0"] - d / 2)) / d
                    frame = blend(render_shot(shots[k - 1], t, rng), render_shot(shots[k], t, rng), f, tin)
                    break
        if frame is None and not shots:
            frame = np.zeros((H, W, 3), np.float32)
        if frame is None:
            cur = [s for s in shots if s["t0"] <= t < s["t1"]]
            shot = cur[-1] if cur else shots[-1]
            frame = apply_transition(render_shot(shot, t, rng), None, shot, t)
        cur_shot = ([s for s in shots if s["t0"] <= t < s["t1"]] or shots[-1:] or [{}])[-1]
        dust = cur_shot.get("dust_cfg", tl.get("dust"))
        if dust:
            frame = frame + dust_layer(t, *dust)
        if shots:
            frame = grade(frame, rng, cur_shot.get("grain_cfg", tl.get("grain")))
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        for line in sub_lines:
            line.draw(frame, t)
        fade = min(1.0, (t - t_start) / tl.get("fade_in", 0.5), (t_end - t) / tl.get("fade_out", 0.8))
        frame = np.clip(frame * ease(fade), 0, 1)
        if a.preview:
            frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
        proc.stdin.write((frame * 255).astype(np.uint8).tobytes())
        if i % (fps * 2) == 0:
            print(f"{t - t_start:5.1f}s / {t_end - t_start:.1f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    print(path)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
