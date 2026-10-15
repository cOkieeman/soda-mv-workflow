"""第一夜 MV：GPU 版合成引擎（PyTorch CUDA ＋ NVENC），結果對齊 compose.py。

  python compose_gpu.py full               # 24fps 全片 → out/full.mp4
  python compose_gpu.py full --fps 60      # 60fps
  python compose_gpu.py s1 --check 70.5    # 只算一格，和 CPU 版比對（輸出對照圖與 PSNR）

GPU 負責：取景／旋轉／深度視差（grid_sample）、交叉淡化與轉場、燈火、光掃、雨、星芒、調色、顆粒、暗角、浮塵模糊。
CPU 負責：讀圖、光流補格（逐格片段）、畫雨絲與浮塵點、字幕疊字（小區塊）。
編碼：h264_nvenc（顯卡硬體編碼）。
"""
import argparse, math, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # ComfyUI 內嵌的 Python 不會自動加入腳本所在資料夾

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import compose as C
import imageio_ffmpeg

ROOT = C.ROOT
W, H = C.W, C.H
DEV = torch.device("cuda")
torch.backends.cudnn.benchmark = True

_tex = {}          # 路徑 → GPU 貼圖（1,3,h,w）float16，BGR
_tex_order = []    # 逐格素材的 LRU
_grid_cache = {}
_kern = {}


# ── 基本工具 ─────────────────────────────────────────────
def to_gpu(img):
    """numpy HxWx3 float(0–1, BGR) → (1,3,H,W) float32 cuda"""
    return torch.from_numpy(np.ascontiguousarray(img)).to(DEV, non_blocking=True).permute(2, 0, 1)[None].float()


def tex(path):
    """讀圖並快取在 GPU（float16 省顯存）。逐格素材只留最近 120 張。"""
    path = C.hd(path)
    if path not in _tex:
        im = cv2.imdecode(np.fromfile(str(ROOT / path), np.uint8), cv2.IMREAD_COLOR)
        _tex[path] = torch.from_numpy(im).to(DEV).permute(2, 0, 1)[None].half() / 255.0
        if "frame_" in path.replace("\\", "/"):
            _tex_order.append(path)
            while len(_tex_order) > 120:
                _tex.pop(_tex_order.pop(0), None)
    return _tex[path].float()


def gkernel(sigma):
    if sigma not in _kern:
        r = int(math.ceil(sigma * 3))
        x = torch.arange(-r, r + 1, device=DEV, dtype=torch.float32)
        k = torch.exp(-(x / sigma) ** 2 / 2)
        _kern[sigma] = k / k.sum()
    return _kern[sigma]


def gblur(x, sigma):
    """可分離高斯模糊；大 sigma 先縮小再模糊再放大（視覺上一致、快很多）。x：(1,C,H,W)"""
    if sigma <= 0:
        return x
    scale = 1
    while sigma / scale > 8 and min(x.shape[-2:]) // (scale * 2) > 16:
        scale *= 2
    y = F.avg_pool2d(x, scale) if scale > 1 else x
    k = gkernel(sigma / scale)
    c = y.shape[1]
    r = (k.numel() - 1) // 2
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    if scale > 1:
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return y


def uv01():
    if "uv01" not in _grid_cache:
        v, u = torch.meshgrid(torch.linspace(0, 1, H, device=DEV), torch.linspace(0, 1, W, device=DEV), indexing="ij")
        _grid_cache["uv01"] = (u, v)
    return _grid_cache["uv01"]


def col(c):
    return torch.tensor([c[2] / 255.0, c[1] / 255.0, c[0] / 255.0], device=DEV).view(1, 3, 1, 1)


# ── 取景 ────────────────────────────────────────────────
def sample(img, mx, my):
    """img (1,3,h,w)，mx/my 為來源像素座標（H,W）→ 雙線性取樣，邊界反射（同 cv2.remap BORDER_REFLECT）。"""
    h, w = img.shape[-2:]
    gx = (mx + 0.5) / w * 2 - 1
    gy = (my + 0.5) / h * 2 - 1
    grid = torch.stack([gx, gy], -1)[None]
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def camera(img, z, c, depth=None, par=0.0, z0=1.0, rot=0.0):
    h, w = img.shape[-2:]
    s = max(W / w, H / h) * z
    if "pix" not in _grid_cache:
        v, u = torch.meshgrid(torch.arange(H, device=DEV, dtype=torch.float32),
                              torch.arange(W, device=DEV, dtype=torch.float32), indexing="ij")
        _grid_cache["pix"] = (u - W / 2, v - H / 2)
    du, dv = _grid_cache["pix"]
    cx, cy = c[0] * w, c[1] * h
    cx = min(max(cx, W / s / 2), w - W / s / 2)
    cy = min(max(cy, H / s / 2), h - H / s / 2)
    if rot:
        a = math.radians(rot)
        du, dv = du * math.cos(a) - dv * math.sin(a), du * math.sin(a) + dv * math.cos(a)
    mx = cx + du / s
    my = cy + dv / s
    if depth is not None and par:
        d = sample(depth, mx, my)[0, 0]
        k = 1.0 + par * (z - z0) * (d - 0.5)
        mx = cx + (mx - cx) / k
        my = cy + (my - cy) / k
    out = sample(img, mx, my)
    to_out = lambda x, y: (((x * w - cx) * s + W / 2) / W, ((y * h - cy) * s + H / 2) / H)
    return out, to_out, s


def depth_tex(path, shape):
    key = ("depth", path, shape)
    if key not in _tex:
        d = C.load_depth(path, shape)          # CPU 版已做好縮放與模糊
        _tex[key] = torch.from_numpy(d).to(DEV)[None, None]
    return _tex[key]


def base_image(shot, t):
    if "seq" in shot:
        fs = C.seq_frames(shot["seq"])
        if shot.get("frames"):
            fs = fs[shot["frames"][0] - 1:shot["frames"][1]]
        q = min(max((t - shot["t0"]) / (shot["t1"] - shot["t0"]), 0.0), 1.0)
        if shot.get("pingpong"):
            q = 1 - abs(1 - 2 * q)
        if shot.get("timemap"):  # 分段變速：[(鏡頭內秒數, 素材秒數), ...]，素材以 24fps 計，中間線性內插
            tt = (t - shot["t0"])
            km = shot["timemap"]
            src_s = km[-1][1]
            for (a0, b0), (a1, b1) in zip(km, km[1:]):
                if tt <= a1:
                    src_s = b0 + (b1 - b0) * max(0.0, min(1.0, (tt - a0) / (a1 - a0)))
                    break
            q = min(max(src_s * 24 / (len(fs) - 1), 0.0), 1.0)
        x = q * (len(fs) - 1)
        i = int(x)
        # 預設直接取最近的原始格：AI 動畫格間動作大，光流補格會猜錯、把兩格疊成拖影（使用者 10-15 回饋）
        if shot.get("interp") != "flow":
            i = min(int(round(x)), len(fs) - 1)
        if shot.get("interp") == "flow" and i + 1 < len(fs) and x - i > 0.02:
            a = C.load(str(fs[i].relative_to(ROOT)))
            b = C.load(str(fs[i + 1].relative_to(ROOT)))
            img = to_gpu(C.flow_interp(a, b, x - i))           # 光流仍在 CPU
            C._cache.pop(C.hd(str(fs[i].relative_to(ROOT))), None)
        else:
            img = tex(str(fs[i].relative_to(ROOT)))
        if shot.get("resize"):
            rw, rh = shot["resize"]
            img = F.interpolate(img, size=(rh, rw), mode="bicubic", align_corners=False)
        return img
    imgs = [(shot["t0"], shot["src"])] + list(shot.get("xfade", []))
    cur = imgs[0][1]
    for (ta, pa), (tb, pb) in zip(imgs, imgs[1:]):
        if t >= tb:
            cur = pb
    xd = shot.get("xdur", 1.2)
    for tb, pb in imgs[1:]:
        if tb - xd / 2 <= t < tb + xd / 2:
            prev = [p for ta, p in imgs if ta < tb][-1]
            a, b = tex(prev), tex(pb)
            if a.shape != b.shape:
                b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
            f = C.ease((t - (tb - xd / 2)) / xd)
            return a * (1 - f) + b * f
    return tex(cur)


def render_shot(shot, t):
    p = (t - shot["t0"]) / (shot["t1"] - shot["t0"])
    p = min(max(p, -0.3), 1.3)
    if shot.get("ease_in"):
        p = p * p * (2 - p) if 0 <= p <= 1 else p
    if shot.get("black"):
        return torch.zeros(1, 3, H, W, device=DEV)
    img = base_image(shot, t)
    z0, z1, c0, c1 = shot.get("cam", (1.0, 1.08, (0.5, 0.5), (0.5, 0.5)))
    z = max(1.0, z0 + (z1 - z0) * p)
    c = (c0[0] + (c1[0] - c0[0]) * p, c0[1] + (c1[1] - c0[1]) * p)
    depth = depth_tex(shot["depth"], tuple(img.shape[-2:])) if shot.get("depth") else None
    r0, r1 = shot.get("rot", (0.0, 0.0))
    out, to_out, scale = camera(img, z, c, depth, shot.get("par", 0.0), z0, r0 + (r1 - r0) * p)
    for fx in shot.get("fx", []):
        out = apply_fx(out, fx, t, shot, to_out)
    return out


# ── 特效 ────────────────────────────────────────────────
def apply_fx(out, fx, t, shot, to_out):
    kind = fx[0]
    u, v = uv01()
    if kind == "flicker":
        _, x, y, r, c, k = fx
        x, y = to_out(x, y)
        n = 0.75 + 0.25 * math.sin(t * 7.3) * math.sin(t * 2.9 + 1) + 0.08 * math.sin(t * 23.0)
        g = torch.exp(-(((u - x) * 16 / 9) ** 2 + (v - y) ** 2) / (r * r))[None, None]
        return out + g * col(c) * k * n
    if kind == "sweep":
        _, side, c, k, ta, tb = fx
        f = C.ease((t - ta) / (tb - ta))
        ramp = ((u if side == "right" else 1 - u) ** 1.6)[None, None]
        return out * (1 + ramp * f * k * 0.6) + ramp * f * k * col(c) * 0.25
    if kind == "rain":
        _, n, k = fx
        layer = np.zeros((H, W), np.float32)
        pts = np.random.default_rng(5).random((n, 3))
        for px, py, zz in pts:
            spd, ln = 900 + 1500 * zz, 18 + 50 * zz
            x = (px * (W + 300) + t * spd * 0.18) % (W + 300) - 150
            y = (py * (H + 200) + t * spd) % (H + 200) - 100
            cv2.line(layer, (int(x), int(y)), (int(x - ln * 0.18), int(y - ln)), 0.25 + 0.6 * zz, 1, cv2.LINE_AA)
        L = gblur(torch.from_numpy(layer).to(DEV)[None, None], 0.8)
        rc = torch.tensor([0.92, 0.9, 0.85], device=DEV).view(1, 3, 1, 1)
        return out * (1 - L * k * 0.25) + L * k * rc
    if kind == "inkrain":  # 深褐細墨線雨：疏、細、慢，壓暗疊色，配舊紙墨色畫面
        _, n, k = fx
        layer = np.zeros((H, W), np.float32)
        pts = np.random.default_rng(7).random((n, 3))
        for px, py, zz in pts:
            spd, ln = 500 + 700 * zz, 30 + 60 * zz
            x = (px * (W + 300) + t * spd * 0.12) % (W + 300) - 150
            y = (py * (H + 200) + t * spd) % (H + 200) - 100
            cv2.line(layer, (int(x), int(y)), (int(x - ln * 0.12), int(y - ln)), 0.5 + 0.5 * zz, 1 + int(zz > 0.7), cv2.LINE_AA)
        L = gblur(torch.from_numpy(layer).to(DEV)[None, None], 0.6)
        ic = torch.tensor([0.12, 0.20, 0.30], device=DEV).view(1, 3, 1, 1) # BGR：深褐
        return out * (1 - L * k * 0.9) + L * k * 0.9 * ic
    if kind == "waterdrops":  # 水滴落在舊紙上暈開：圈內墨線泡糊、紙色變深，外緣留一道不規則水痕
        _, drops, spread = fx           # drops: [(落下秒, 圖上 x, 圖上 y, 最大半徑)]；spread: (開始秒, 結束秒) 或 None
        n = ink_noise()
        u, v = uv01()
        wet = torch.zeros_like(u)[None, None]
        tide = torch.zeros_like(wet)
        glint = torch.zeros_like(wet)
        for td, x, y, rmax in drops:
            if t < td:
                continue
            dt = t - td
            r = rmax * (1 - math.exp(-dt / 0.55))
            if spread and t > spread[0]:
                r += C.ease(min((t - spread[0]) / (spread[1] - spread[0]), 1.0)) * 1.6
            ox, oy = to_out(x, y)
            d = torch.sqrt(((u - ox) * 16 / 9) ** 2 + (v - oy) ** 2)[None, None]
            e = r - d + n * 0.035 * min(dt / 0.3, 1.0)
            wet = torch.maximum(wet, (e / 0.02).clamp(0, 1))
            fade = 1 - C.ease(min(max((t - spread[0]) / 0.8, 0.0), 1.0)) if spread else 1.0   # 整片暈開時水痕跟著化掉
            tide = torch.maximum(tide, torch.exp(-(e / 0.012) ** 2) * min(dt / 0.4, 1.0) * fade)
            glint = glint + torch.exp(-(d / 0.012) ** 2) * math.exp(-((dt - 0.05) / 0.08) ** 2)
        if float(wet.max()) <= 0 and float(glint.max()) <= 0:
            return out
        sig = 5.0
        if spread and t > spread[0]:
            sig += C.ease(min((t - spread[0]) / (spread[1] - spread[0]), 1.0)) * 10
        soft = gblur(out, sig)
        damp = soft * 0.88 + gblur(out, sig * 2.5) * 0.12      # 墨被水泡開往外滲
        damp = damp * 0.9 + torch.tensor([0.02, 0.025, 0.03], device=DEV).view(1, 3, 1, 1)  # 濕紙略深略暖（BGR）
        o = out * (1 - wet) + damp * wet
        ink = torch.tensor([0.10, 0.14, 0.19], device=DEV).view(1, 3, 1, 1)
        o = o * (1 - tide * 0.45) + ink * tide * 0.45
        return o + glint.clamp(0, 1) * 0.5
    if kind == "tears":  # 淚滴打濕回憶的畫：淚珠先停在紙上（透明、反光、微放大），再慢慢滲進紙裡，墨被泡開往外滲
        _, drops = fx                   # drops: [(落下秒, 圖上 x, 圖上 y, 最大半徑)]
        n = ink_noise()
        u, v = uv01()
        o = out
        bleed = torch.minimum(out, gblur(out, 12) * 1.02) * 0.6 + gblur(out, 7) * 0.4   # 墨往外滲（暗部擴散）
        for td, x, y, rmax in drops:
            if t < td:
                continue
            dt = t - td
            ox, oy = to_out(x, y)
            dx, dy = (u - ox) * 16 / 9, v - oy
            d = torch.sqrt(dx ** 2 + dy ** 2)[None, None]
            soak = C.ease(min(max((dt - 0.45) / 2.2, 0.0), 1.0))                    # 停半秒後才開始滲
            r = 0.012 + rmax * soak
            e = r - d + n * 0.05 * soak
            wet = (e / 0.035).clamp(0, 1) * min(dt / 0.1, 1.0)                       # 邊緣很柔
            o = o * (1 - wet) + (bleed * 0.85 + torch.tensor([0.01, 0.015, 0.02], device=DEV).view(1, 3, 1, 1)) * wet
            edge = torch.exp(-(e / 0.018) ** 2) * soak * 0.35                         # 淡淡的乾痕
            o = o * (1 - edge) + torch.tensor([0.12, 0.16, 0.2], device=DEV).view(1, 3, 1, 1) * edge
            bead = math.exp(-max(dt - 0.45, 0.0) / 0.35) * min(dt / 0.06, 1.0)          # 淚珠：先在，再被紙吸掉
            if bead > 0.01:
                rb = 0.032
                inb = (1 - d / rb).clamp(0, 1)
                mag = 1 - 0.3 * inb[0, 0]                                             # 水珠中央放大底下的畫
                mx = (ox + (u - ox) * mag) * W - 0.5
                my = (oy + (v - oy) * mag) * H - 0.5
                lens = sample(o, mx, my)
                m = (inb * 4).clamp(0, 1) * bead
                hl = torch.exp(-(((dx + 0.01) ** 2 + (dy + 0.01) ** 2) / 0.005 ** 2))[None, None] * bead
                rim = torch.exp(-((d - rb) / 0.004) ** 2) * bead * 0.5
                o = o * (1 - m) + (lens * 1.04) * m
                o = o * (1 - rim) + torch.tensor([0.08, 0.1, 0.12], device=DEV).view(1, 3, 1, 1) * rim + hl * 0.8
        return o
    if kind == "sparkle":
        _, x, y, ts = fx
        x, y = to_out(x, y)
        a = math.exp(-((t - ts) / 0.18) ** 2)
        if a < 0.01:
            return out
        star = np.zeros((H, W), np.float32)
        cx, cy = int(x * W), int(y * H)
        cv2.line(star, (cx - 26, cy), (cx + 26, cy), 1.0, 1, cv2.LINE_AA)
        cv2.line(star, (cx, cy - 16), (cx, cy + 16), 1.0, 1, cv2.LINE_AA)
        st = torch.from_numpy(star).to(DEV)[None, None]
        st = gblur(st, 2) * 1.4 + gblur(st, 9) * 1.6
        return out + st * a
    # 其他（dream、dust、petals：目前片中沒用到）退回 CPU 版
    o = C.apply_fx(out[0].permute(1, 2, 0).cpu().numpy(), fx, t, shot, np.random.default_rng(1), to_out)
    return to_gpu(o)


def dust_layer(t, n, c):
    if "dust_pts" not in _grid_cache or len(_grid_cache["dust_pts"]) != n:
        _grid_cache["dust_pts"] = np.random.default_rng(11).random((n, 5))
    layer = np.zeros((H, W), np.float32)
    for px, py, sp, ph, sz in _grid_cache["dust_pts"]:
        x = int(((px + 0.010 * t * (0.4 + sp) + 0.02 * math.sin(t * 0.3 + ph * 6)) % 1) * W)
        y = int(((py - 0.016 * t * (0.3 + sp)) % 1) * H)
        a = (0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * (0.8 + 1.6 * sp) + ph * 6.28))) * (0.4 + 0.6 * sz)
        cv2.circle(layer, (x, y), 1 + int(sz * 2.4), a, -1, cv2.LINE_AA)
    L = gblur(torch.from_numpy(layer).to(DEV)[None, None], 2.0)
    return L * col(c)


_noise = None


def grade(img, rng, grain=None):
    global _noise
    gray = img.mean(1, keepdim=True)
    img = gray + (img - gray) * 0.9
    img = img.clamp(min=0) ** 1.06
    if "vig" not in _grid_cache:
        v, u = torch.meshgrid(torch.linspace(-1, 1, H, device=DEV), torch.linspace(-1, 1, W, device=DEV), indexing="ij")
        _grid_cache["vig"] = (1 - 0.28 * (u * u * 0.6 + v * v * 0.9 - 0.15).clamp(0, 1))[None, None]
    img = img * _grid_cache["vig"]
    if _noise is None:
        g = np.random.default_rng(7)
        _noise = [F.interpolate(torch.from_numpy(g.normal(0, 1, (H // 2, W // 2)).astype(np.float32)).to(DEV)[None, None],
                                size=(H, W), mode="bilinear", align_corners=False) for _ in range(6)]
    n = _noise[int(rng.integers(6))]
    return (img + n * (C.GRAIN if grain is None else grain)).clamp(0, 1)


# ── 轉場 ────────────────────────────────────────────────
def luminance(img):
    w = torch.tensor([0.114, 0.587, 0.299], device=DEV).view(1, 3, 1, 1)
    return gblur((img * w).sum(1, keepdim=True), 9)


_ink_noise = None


def ink_noise():
    """墨跡邊緣用的低頻雜訊（兩層，固定種子），-1～1。"""
    global _ink_noise
    if _ink_noise is None:
        g = torch.Generator(device="cpu").manual_seed(11)
        a = torch.rand(1, 1, 18, 32, generator=g)
        b = torch.rand(1, 1, 72, 128, generator=g)
        up = lambda x: torch.nn.functional.interpolate(x.to(DEV), size=(H, W), mode="bicubic", align_corners=False)
        nz = up(a) * 0.7 + up(b) * 0.3
        _ink_noise = (nz - nz.mean()) / (nz.std() * 2.5)
    return _ink_noise


def blend(prev, cur, f, tin):
    kind = tin[0]
    if kind in ("luma", "luma_inv"):
        soft = 0.35
        L = luminance(prev)
        if kind == "luma_inv":
            L = 1 - L
        m = ((f * (1 + soft) - L) / soft).clamp(0, 1)
        return prev * (1 - m) + cur * m
    if kind == "bloom":
        _, d, x, y, c = tin
        u, v = uv01()
        dist = torch.sqrt(((u - x) * 16 / 9) ** 2 + (v - y) ** 2)[None, None]
        r = C.ease(f) * 2.4
        m = ((r - dist) / 0.6).clamp(0, 1)
        glow = torch.exp(-(dist / (0.15 + r * 0.5)) ** 2) * math.sin(math.pi * f) * 0.55
        return prev * (1 - m) + cur * m + glow * col(c)
    if kind == "ink":  # 墨跡暈染：幾團墨從畫面中心向外滲開，邊緣不規則、帶深褐墨暈，墨到之處露出新畫面
        n = ink_noise()
        u, v = uv01()
        field = None
        for x, y, delay, rmax in ((0.42, 0.48, 0.0, 1.4), (0.18, 0.28, 0.2, 0.75), (0.8, 0.72, 0.3, 0.75)):
            g = C.ease(min(max((f - delay) / (1 - delay), 0.0), 1.0))
            d = torch.sqrt(((u - x) * 16 / 9) ** 2 + (v - y) ** 2)[None, None]
            fi = g * rmax - d
            field = fi if field is None else torch.maximum(field, fi)
        e = field + n * 0.16 * (0.4 + 0.6 * math.sin(math.pi * min(f * 1.2, 1.0)))
        m = (e / 0.05 + 0.5).clamp(0, 1)
        m = torch.maximum(m, torch.tensor(C.ease(min(max((f - 0.85) / 0.15, 0.0), 1.0)), device=DEV))
        rim = torch.exp(-(e / 0.035) ** 2) * (1 - f) ** 0.6 * 0.6
        halo = torch.exp(-((e + 0.05) / 0.07) ** 2) * (1 - f) * 0.35        # 墨水往外滲的淡暈
        ink = torch.tensor([0.06, 0.09, 0.13], device=DEV).view(1, 3, 1, 1)  # BGR：深褐墨
        o = prev * (1 - m) + cur * m
        o = o * (1 - halo) + ink * halo
        return o * (1 - rim) + ink * rim
    if kind == "dim":
        g = C.ease(f)
        return (prev * (1 - g) + cur * g) * (1 - 0.5 * math.sin(math.pi * f))
    f = C.ease(f)
    return prev * (1 - f) + cur * f


def apply_transition(frame, shot, t):
    tin = shot.get("tin", ("cut",))
    dt = t - shot["t0"]
    if tin[0] == "flash":
        f = math.exp(-(dt / (tin[1] * 0.4)) ** 2) * 0.42
        return frame * (1 - f * 0.4) + col(tin[2]) * f
    if tin[0] == "dip":
        return frame * C.ease(min(1.0, dt / tin[1]))
    return frame


def render_frame(tl, shots, t, rng):
    frame = None
    for k in range(1, len(shots)):
        tin = shots[k].get("tin", ("cut",))
        if tin[0] in ("dissolve", "luma", "luma_inv", "bloom", "dim", "ink"):
            d = tin[1]
            if shots[k]["t0"] - d / 2 <= t < shots[k]["t0"] + d / 2:
                f = (t - (shots[k]["t0"] - d / 2)) / d
                frame = blend(render_shot(shots[k - 1], t), render_shot(shots[k], t), f, tin)
                break
    if frame is None and not shots:
        frame = torch.zeros(1, 3, H, W, device=DEV)
    if frame is None:
        cur = [s for s in shots if s["t0"] <= t < s["t1"]]
        shot = cur[-1] if cur else shots[-1]
        frame = apply_transition(render_shot(shot, t), shot, t)
    cur_shot = ([s for s in shots if s["t0"] <= t < s["t1"]] or shots[-1:] or [{}])[-1]
    dust = cur_shot.get("dust_cfg", tl.get("dust"))
    if dust:
        frame = frame + dust_layer(t, *dust)
    if shots:
        frame = grade(frame, rng, cur_shot.get("grain_cfg", tl.get("grain")))
    return frame


def main():
    from timelines import TIMELINES
    import subs as subs_mod
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out")
    ap.add_argument("--check", type=float, help="只算這一秒，和 CPU 版比對")
    ap.add_argument("--cq", type=int, default=16, help="NVENC 品質（越小越好）")
    a = ap.parse_args()
    tl = TIMELINES[a.name]
    shots, t_start, t_end = tl["shots"], tl["t0"], tl["t1"]
    sub_lines = subs_mod.lines_between(t_start, t_end, tl.get("sub_pos")) if tl.get("audio") else []
    sub_lines += [subs_mod.Title(*ti) for ti in tl.get("titles", [])]
    sub_lines += [subs_mod.Card(*c) for c in tl.get("cards", [])]

    def finish(frame_gpu, t):
        fr = frame_gpu[0].permute(1, 2, 0).float().cpu().numpy()
        fr = np.ascontiguousarray(fr)
        for line in sub_lines:
            line.draw(fr, t)
        fade = min(1.0, (t - t_start) / tl.get("fade_in", 0.5), (t_end - t) / tl.get("fade_out", 0.8))
        return np.clip(fr * C.ease(fade), 0, 1)

    if a.check is not None:   # GPU 與 CPU 同一格對照
        t = a.check
        g = finish(render_frame(tl, shots, t, np.random.default_rng(3)), t)
        rng = np.random.default_rng(3)
        cur = [s for s in shots if s["t0"] <= t < s["t1"]][-1]
        c = C.render_shot(cur, t, rng)
        c = C.grade(c + (C.dust_layer(t, *cur.get("dust_cfg", tl.get("dust"))) if (cur.get("dust_cfg") or tl.get("dust")) else 0), np.random.default_rng(3), cur.get("grain_cfg", tl.get("grain")))
        c = np.ascontiguousarray(c, dtype=np.float32)
        for line in sub_lines:
            line.draw(c, t)
        mse = float(((g - c) ** 2).mean())
        print(f"PSNR {10 * math.log10(1 / max(mse, 1e-10)):.1f} dB")
        both = np.hstack([cv2.resize(c, (960, 540)), cv2.resize(g, (960, 540))])
        cv2.imencode(".jpg", (both * 255).astype(np.uint8))[1].tofile(str(ROOT / "out" / "_gpu_check.jpg"))
        return

    fps = a.fps
    path = Path(a.out) if a.out else ROOT / "out" / f"{a.name}{'' if fps == 24 else f'_{fps}fps'}.mp4"
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-"] + (["-ss", str(t_start), "-t", str(t_end - t_start), "-i", str(ROOT / tl["audio"])] if tl.get("audio")
           else ["-ss", str(tl["bgm"][1]), "-t", str(t_end - t_start), "-i", str(ROOT / tl["bgm"][0])] if tl.get("bgm")
           else ["-f", "lavfi", "-t", str(t_end - t_start), "-i", "anullsrc=r=48000:cl=stereo"]) + [
           "-af", (f"volume={tl['bgm'][2]}dB,afade=t=in:d={tl['bgm'][3]},afade=t=out:st={t_end - t_start - tl['bgm'][4]}:d={tl['bgm'][4]}"
                   if tl.get("bgm") and not tl.get("audio") else f"afade=t=in:d=0.4,afade=t=out:st={t_end - t_start - 0.8}:d=0.8"),
           "-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", str(a.cq), "-b:v", "0",
           "-profile:v", "high", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rng = np.random.default_rng(7)
    n = int(round((t_end - t_start) * fps))
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            t = t_start + i / fps
            fr = finish(render_frame(tl, shots, t, rng), t)
            proc.stdin.write((fr * 255).astype(np.uint8).tobytes())
            if i % (fps * 5) == 0:
                el = time.time() - t0
                print(f"{t - t_start:6.1f}s / {t_end - t_start:.1f}s   {el / max(i, 1):.3f} s/格", flush=True)
    proc.stdin.close()
    proc.wait()
    print(path, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
