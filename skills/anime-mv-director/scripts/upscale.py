"""第一夜 MV：用 SeedVR2（與使用者 workflow 相同的 7B sharp fp8）把逐格短片放大到 1080p。

  python upscale.py clips/13_CG11_1_101 [--res 1080] [--batch 21]
輸出：<clip>/hd/frame_00001.png …
"""
import argparse, sys, urllib.parse, urllib.request
from pathlib import Path

from gen import run

HOST = "http://127.0.0.1:8190"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--res", type=int, default=1080, help="短邊目標像素")
    ap.add_argument("--batch", type=int, default=9, help="4n+1；越大時間一致性越好、越吃顯存")
    ap.add_argument("--cap", type=int, default=0, help="只放大前 N 格（0＝全部）")
    a = ap.parse_args()
    clip = Path(a.clip).resolve()
    # 16GB 顯存：blocks_to_swap 32、小 batch、VAE 分塊 512，避免頂滿觸發驅動錯誤
    wf = {
        "1": {"class_type": "VHS_LoadImagesPath", "inputs": {"directory": str(clip), "image_load_cap": a.cap,
                                                              "skip_first_images": 0, "select_every_nth": 1}},  # image_load_cap 由 --cap 決定
        "2": {"class_type": "SeedVR2LoadDiTModel", "inputs": {
            "model": "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors", "device": "cuda:0",
            "blocks_to_swap": 32, "swap_io_components": False, "offload_device": "cpu", "cache_model": False,
            "attention_mode": "sageattn_2"}},
        "3": {"class_type": "SeedVR2LoadVAEModel", "inputs": {
            "model": "ema_vae_fp16.safetensors", "device": "cuda:0", "encode_tiled": True, "encode_tile_size": 512,
            "encode_tile_overlap": 64, "decode_tiled": True, "decode_tile_size": 512, "decode_tile_overlap": 64,
            "tile_debug": "false", "offload_device": "cpu", "cache_model": False}},
        "4": {"class_type": "SeedVR2VideoUpscaler", "inputs": {
            "image": ["1", 0], "dit": ["2", 0], "vae": ["3", 0], "seed": 42, "resolution": a.res, "max_resolution": 1920,
            "batch_size": a.batch, "uniform_batch_size": True, "color_correction": "lab", "temporal_overlap": 4,
            "prepend_frames": 0, "input_noise_scale": 0, "latent_noise_scale": 0, "offload_device": "cpu",
            "enable_debug": False}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": f"mv_hd/{clip.name}"}},
    }
    out = run(wf)
    hd = clip / "hd"
    hd.mkdir(exist_ok=True)
    for i, img in enumerate(out["5"]["images"], 1):
        q = urllib.parse.urlencode({k: img[k] for k in ("filename", "subfolder", "type")})
        (hd / f"frame_{i:05d}.png").write_bytes(urllib.request.urlopen(f"{HOST}/view?{q}").read())
    print(hd, len(out["5"]["images"]), "frames")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
