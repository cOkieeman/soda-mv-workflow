"""第一夜 MV：替圖片產生 2.5D 用的分層素材（透過 ComfyUI）。

  python layers.py 背景CG/CG11_3.png renders/niu/12_1.png
輸出：layers/<檔名>/mask.png（人物去背遮罩，BiRefNet-HR-matting）、depth.png（Depth Anything V2，越亮越近）
"""
import json, sys, urllib.parse, urllib.request
from pathlib import Path

from gen import run, upload

ROOT = Path(__file__).parent
HOST = "http://127.0.0.1:8190"


def fetch(img):
    q = urllib.parse.urlencode({k: img[k] for k in ("filename", "subfolder", "type")})
    return urllib.request.urlopen(f"{HOST}/view?{q}", timeout=60).read()


def make(path):
    name = upload(path)
    wf = {
        "1": {"class_type": "LoadImage", "inputs": {"image": name}},
        "2": {"class_type": "BiRefNetRMBG", "inputs": {"image": ["1", 0], "model": "BiRefNet-HR-matting",
                                                         "sensitivity": 1.0, "mask_blur": 0, "mask_offset": 0,
                                                         "invert_output": False, "refine_foreground": False,
                                                         "unload_model": True, "background": "Alpha",
                                                         "background_color": "#222222"}},
        "3": {"class_type": "MaskToImage", "inputs": {"mask": ["2", 1]}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "mv_layers/mask"}},
        "5": {"class_type": "DepthAnythingV2Preprocessor", "inputs": {"image": ["1", 0],
                                                                       "ckpt_name": "depth_anything_v2_vitl.pth",
                                                                       "resolution": 1024}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": "mv_layers/depth"}},
    }
    out = run(wf)
    d = ROOT / "layers" / Path(path).stem
    d.mkdir(parents=True, exist_ok=True)
    (d / "mask.png").write_bytes(fetch(out["4"]["images"][0]))
    (d / "depth.png").write_bytes(fetch(out["6"]["images"][0]))
    print(d)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for p in sys.argv[1:]:
        make(p)
