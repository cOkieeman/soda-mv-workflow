"""第一夜 MV：用 Codex（GPT 生圖）生成 src=codex 的鏡頭。

用法：
  python codex_gen.py 52 20 --n 2      # 指定鏡頭
  python codex_gen.py --all --n 2      # 所有 src=codex 的鏡頭
輸出：renders/codex/<鏡號>_<n>.png
"""
import argparse, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
CODEX = Path.home() / "AppData/Local/OpenAI/Codex/bin/9614f1d9b40d9c4f/codex.exe"
REF = ROOT / "原立绘"
CG = ROOT / "背景CG"

# 造型：立繪參考圖＋文字描述
LOOKS = {
    "liuxiang_eye": (REF / "柳巷 (2)/角色04_622x1728_14张/55通常.png",
                     "long side-swept bangs covering one eye, small side bun, very long dark brown hair, "
                     "plain grey-green hanfu robe with faint white floral embroidery, pale long skirt"),
    "liuxiang": (REF / "柳巷/角色08_580x1728_10张/69微笑.png",
                 "bangs parted so both eyes are visible, a small white flower hairpin with gold tassels, "
                 "very long dark brown hair, plain grey-green hanfu robe with faint white floral embroidery, pale long skirt"),
    "huajie": (REF / "花街/角色09_770x1709_9张/85平静.png",
               "blunt bangs, hair up with a gold ornament and tassels, pearl necklace, dark green outer robe "
               "with white patterned trim, white long skirt"),
    "toupai": (REF / "头牌/角色03_967x2160_15张/40通常.png",
               "high bun with gold hairpins and tassels, ornate gold necklace, rich dark green layered hanfu "
               "with gold accessories, a beautiful famous courtesan"),
    "fangzhiyou": (REF / ".." / "原立绘" / "方知宥_立繪.png",
                   "young Chinese scholar Fang Zhiyou: pale slender face, calm narrow dark eyes, very long straight dark brown "
                   "hair with a thin dark headband tied with long pale grey ribbons, white cross-collar inner robe, grey-black "
                   "wide-sleeved outer robe with gold cloud-pattern trim on the cuffs, dark purple sash, holds a folding fan"),
}
# 鏡號 → (造型, 畫風參考 CG)
PLAN = {
    "11": ("liuxiang", "cg31_1"), "17": ("liuxiang", "cg31_1"), "20": ("liuxiang", "cg31_2"),
    "23": ("liuxiang", "cg14_1"), "24": ("huajie", "cg54_1"), "26": ("huajie", "cg54_1"),
    "43a": ("huajie", "cg54_1"), "43b": ("liuxiang", "cg13_1"), "43c": ("huajie", "cg31_1"),
    "43d": ("toupai", "cg30_1"), "51": ("toupai", "cg49_1"), "52": ("liuxiang_eye", "CG11_1"),
    # 香囊特寫：沒有她出場，只附畫風 CG＋香囊原設定
    "05": ("fangzhiyou", "cg47_1"), "08": ("fangzhiyou", "cg47_1"), "12": ("fangzhiyou", "cg14_1"),
    "35": ("fangzhiyou", "cg54_1"), "36": ("fangzhiyou", "cg54_1"), "37": ("fangzhiyou", "cg54_1"),
    "38": ("fangzhiyou", "cg47_1"), "40": ("fangzhiyou", "cg49_3"), "41": ("fangzhiyou", "cg13_1"),
    "45": (None, "cg47_1"), "47": ("fangzhiyou", "cg47_1"), "49": ("fangzhiyou", "桥上幻景5"),
    "53": ("liuxiang_eye", "cg14_1"),
    "01": (None, "cg47_1"), "03": (None, "cg13_1"), "18": (None, "cg31_1"),
    "41": (None, "cg13_1"), "42": (None, "cg13_1"), "P3": (None, "cg31_1"),
}

PROMPT_NOCHAR = """Use your built-in image generation tool (not code, not scripts) to create exactly ONE image.

Format: wide landscape 16:9, cinematic still for an anime music video.
Style: Reference 1 is the style reference CG. Match it exactly — the same painterly semi-realistic anime illustration
of the visual novel "哀鴻：城破十日記": soft brush texture, muted desaturated palette, deep shadows, cinematic lighting.
Late Ming / early Qing dynasty China setting.
Scene: {scene}
Strictly: no text, no letters, no signature, no watermark, no frame, no split panels.

When done, copy the generated PNG exactly to: {out}
Reply only with the saved path."""

PROMPT = """Use your built-in image generation tool (not code, not scripts) to create exactly ONE image.

Format: wide landscape 16:9, cinematic still for an anime music video.
Style: match the attached style-reference CG exactly — the same painterly semi-realistic anime illustration of the
visual novel "哀鴻：城破十日記": soft brush texture, muted desaturated palette, deep shadows, cinematic lighting.
Main character: the young Chinese woman shown in the attached references.
- Reference 1 is her FACE close-up: the face must match it exactly — slender oval face, small pointed chin, large
  slightly drooping almond-shaped light brown eyes with dark upper lashes and faint shadows under the eyes, pale skin,
  small thin-lipped mouth, quiet melancholic look. Do not make her more generic, older, rounder-faced or more doll-like.
- Reference 2 is her full outfit: {look}.
- Reference 3 is the style reference CG. Late Ming dynasty Yangzhou setting. If a young man appears, his face is mostly hidden or turned away.
Scene: {scene}
Strictly: no text, no letters, no signature, no watermark, no frame, no split panels.

When done, copy the generated PNG exactly to: {out}
Reply only with the saved path."""


def gen(sid, i, shot):
    look, cg = PLAN[sid]
    ref, desc = LOOKS[look] if look else (None, "")
    out = ROOT / "renders" / "codex" / f"{sid}_{i}.png"
    scene = shot["p"]
    extra = shot.get("refs", [])  # 道具等額外參考圖（如香囊原設定）
    if extra:
        scene += (f"\nProp reference: Reference {2 if look is None else 4} shows the exact original design of the sachet; "
                  "reproduce its shape, colors, butterfly embroidery, knot and tassel faithfully.")
    if look:
        prompt = PROMPT.format(look=desc, scene=scene, out=out.relative_to(ROOT).as_posix())
        imgs = [ROOT / "原立绘" / "_臉部" / f"{look}.png", ref, CG / f"{cg}.png"]
    else:
        prompt = PROMPT_NOCHAR.format(scene=scene, out=out.relative_to(ROOT).as_posix())
        imgs = [CG / f"{cg}.png"]
    imgs += [ROOT / e for e in extra]
    t = time.time()
    cmd = [str(CODEX), "exec", "--enable", "image_generation", "--skip-git-repo-check",
           "-s", "workspace-write", "-C", str(ROOT), "-"] + sum([["-i", str(p)] for p in imgs], [])
    for attempt in range(5):  # 429 限流時退避重試
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=1200)
        if out.exists() or "429" not in r.stdout + r.stderr:
            break
        time.sleep(90 * (attempt + 1))
    ok = out.exists()
    if ok:
        json.dump({"shot": sid, "look": look, "face_ref": True, "style_cg": cg, "prompt": prompt},
                  open(out.with_suffix(".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tail = "" if ok else " 失敗：" + (r.stdout + r.stderr).strip()[-300:]
    return f"{out.name}  {time.time() - t:.0f}s{tail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    shots = {s["id"]: s for s in json.load(open(ROOT / "storyboard" / "shots.json", encoding="utf-8"))["shots"]}
    ids = [k for k, s in shots.items() if s.get("src") == "codex"] if a.all else a.shots
    (ROOT / "renders" / "codex").mkdir(parents=True, exist_ok=True)
    tasks = []
    for sid in ids:  # 編號接在現有檔案之後，不覆蓋舊圖
        start = max([int(p.stem.split("_")[-1]) for p in (ROOT / "renders" / "codex").glob(f"{sid}_*.png")] or [0])
        tasks += [(sid, start + i, shots[sid]) for i in range(1, a.n + 1)]
    with ThreadPoolExecutor(a.jobs) as ex:
        for msg in ex.map(lambda t: gen(*t), tasks):
            print(msg, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
