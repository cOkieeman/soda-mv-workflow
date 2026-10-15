"""第一夜 MV：定稿圖去噪＋放大。

  python enhance.py renders/niu/12_1.png --mode rtx     # NVIDIA RTX Video Super Resolution（快）
  python enhance.py renders/niu/12_1.png --mode seedvr2 # SeedVR2 7B（慢，修細節／去噪更強）
輸出：<原檔名>_<mode>.png 放在 renders/hd/
"""
import argparse, sys, urllib.parse, urllib.request
from pathlib import Path

from gen import run, upload

ROOT = Path(__file__).parent
HOST = "http://127.0.0.1:8190"


def workflow(name, mode, short=1440):
    wf = {"1": {"class_type": "LoadImage", "inputs": {"image": name}}}
    if mode == "rtx":
        wf["2"] = {"class_type": "RTXVideoSuperResolution", "inputs": {
            "images": ["1", 0], "resize_type": "scale by multiplier", "resize_type.scale": 2.0, "quality": "ULTRA"}}
    else:
        wf["3"] = {"class_type": "SeedVR2LoadDiTModel", "inputs": {
            "model": "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors", "device": "cuda:0",
            "blocks_to_swap": 32, "swap_io_components": False, "offload_device": "cpu", "cache_model": True,
            "attention_mode": "sageattn_2"}}
        wf["4"] = {"class_type": "SeedVR2LoadVAEModel", "inputs": {
            "model": "ema_vae_fp16.safetensors", "device": "cuda:0", "encode_tiled": True, "encode_tile_size": 512,
            "encode_tile_overlap": 64, "decode_tiled": True, "decode_tile_size": 512, "decode_tile_overlap": 64,
            "tile_debug": "false", "offload_device": "cpu", "cache_model": True}}
        wf["2"] = {"class_type": "SeedVR2VideoUpscaler", "inputs": {
            "image": ["1", 0], "dit": ["3", 0], "vae": ["4", 0], "seed": 42, "resolution": short,
            "max_resolution": 3840, "batch_size": 1, "uniform_batch_size": False, "color_correction": "lab",
            "temporal_overlap": 0, "prepend_frames": 0, "input_noise_scale": 0, "latent_noise_scale": 0,
            "offload_device": "cpu", "enable_debug": False}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": f"mv_hd/{mode}"}}
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--mode", choices=["rtx", "seedvr2"], default="seedvr2")
    ap.add_argument("--short", type=int, default=1440, help="SeedVR2 輸出短邊")
    a = ap.parse_args()
    out = ROOT / "renders" / "hd"
    out.mkdir(parents=True, exist_ok=True)
    for p in a.images:
        dst = out / f"{Path(p).stem}_{a.mode}.png"
        if dst.exists():
            print(dst.name, "已存在"); continue
        res = run(workflow(upload(p), a.mode, a.short))
        img = res["9"]["images"][0]
        q = urllib.parse.urlencode({k: img[k] for k in ("filename", "subfolder", "type")})
        dst.write_bytes(urllib.request.urlopen(f"{HOST}/view?{q}", timeout=120).read())
        print(dst.name, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
