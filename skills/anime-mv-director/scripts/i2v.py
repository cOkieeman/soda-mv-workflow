"""第一夜 MV：用本機 MiniMax H3（使用者的 Soda-MiniMax-H3-16g workflow）把一張圖做成短片。

用法：
  python i2v.py 13 背景CG/CG11_1.png "she sobs quietly ..." --sec 5 --mp 0.4
輸出：clips/<鏡號>_<底圖檔名>_<seed>/frame_00001.png …（24fps 逐格），同資料夾 meta.json。
"""
import argparse, copy, json, random, sys, urllib.parse, urllib.request
from pathlib import Path

from gen import api, run, upload

ROOT = Path(__file__).parent
WF = ROOT / "workfl" / "Soda-MiniMax-H3-16g.json"

KEEP = ("Preserve exactly the character, face, painterly anime illustration style, colors, lighting and composition of "
        "<Picture 1>. The camera stays still. Keep all movements small, slow and natural. Do not add new objects or "
        "people, do not change her face or clothes. ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shot")
    ap.add_argument("image")
    ap.add_argument("motion", help="動作描述（英文），會接在保持原圖的固定說明後面")
    ap.add_argument("--sec", type=float, default=5)
    ap.add_argument("--mp", type=float, default=0.4, help="生成解析度（百萬像素），16GB 建議 0.4")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()

    seed = a.seed if a.seed is not None else random.randrange(1, 2**40)
    wf = copy.deepcopy(json.load(open(WF, encoding="utf-8")))
    wf["20"]["inputs"]["image"] = upload(a.image)
    wf["30"]["inputs"]["text"] = KEEP + a.motion
    wf["106"]["inputs"]["seed"] = seed
    wf["116:12"]["inputs"]["value"] = a.sec
    wf["116:11"]["inputs"]["aspect_ratio"] = "16:9 (Widescreen)"
    wf["116:11"]["inputs"]["megapixels"] = a.mp
    wf["116:108"]["inputs"]["value"] = f"MV第一夜_{a.shot}"
    wf["116:109"]["inputs"]["value"] = seed % 1000
    wf["116:10"]["inputs"]["segment"] = 1
    wf["990"] = {"class_type": "SaveImage", "inputs": {"images": ["120:85", 0], "filename_prefix": f"mv_i2v/{a.shot}"}}

    outputs = run(wf)
    out = ROOT / "clips" / f"{a.shot}_{Path(a.image).stem}_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    imgs = outputs["990"]["images"]
    for i, img in enumerate(imgs, 1):
        q = urllib.parse.urlencode({k: img[k] for k in ("filename", "subfolder", "type")})
        (out / f"frame_{i:05d}.png").write_bytes(urllib.request.urlopen(f"http://127.0.0.1:8190/view?{q}").read())
    json.dump({"shot": a.shot, "image": a.image, "motion": a.motion, "sec": a.sec, "mp": a.mp, "seed": seed,
               "frames": len(imgs), "fps": 24}, open(out / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(out, len(imgs), "frames")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
