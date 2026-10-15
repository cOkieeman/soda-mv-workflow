"""第一夜 MV：透過 Niu 中轉（gpt-image-2.5）用多張參考圖生圖。

金鑰讀自 ~/.codex/niu-image-gen-config.json（Niu Image Gen 插件的設定），不會印出。
用法：
  python niu_gen.py 52 20 --n 2           # 指定鏡頭
  python niu_gen.py --src codex --n 2     # shots.json 中 src=codex 的全部鏡頭
輸出：renders/niu/<鏡號>_<n>.png，同名 .json 記錄 prompt 與參考圖。
提示詞與參考圖的組法沿用 codex_gen.py（PLAN / LOOKS / 畫風 CG / refs）。
"""
import argparse, base64, json, random, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from codex_gen import CG, LOOKS, PLAN as CODEX_PLAN, ROOT

# 空景／道具／老人鏡頭：沒有她出場，只依時段氣氛挑畫風參考 CG
DAY, NIGHT, RAIN, DARK, DREAM, NIGHTSKY = "cg47_1", "cg14_1", "cg54_1", "cg15", "桥上幻景5", "cg49_3"
SCENE_PLAN = {
    "05": DAY, "06": "cg31_1", "07": DAY, "08": DAY, "09": DAY, "10": DAY, "12": NIGHT, "16": "cg31_1",
    "21": DAY, "25": RAIN, "27": RAIN, "30": DARK, "33": DARK, "34": RAIN, "35": RAIN, "36": RAIN, "37": RAIN,
    "38": DAY, "39": NIGHTSKY, "40": NIGHTSKY, "44": NIGHTSKY, "45": DAY, "46": DAY, "47": DAY,
    "48": DREAM, "49": DREAM, "50": RAIN,
}
PLAN = {**{k: (None, v) for k, v in SCENE_PLAN.items()}, **CODEX_PLAN}  # CODEX_PLAN 有指定角色的優先

FACE_M = ("Main character: the young man Fang Zhiyou in the references; match his face and outfit exactly: {look}. "
          "He is young (early twenties), never old, no white hair, no beard.")
HIM = ("If the young man Fang Zhiyou appears, he matches his reference sheet (Reference: his full outfit): very long dark "
       "hair, thin headband with long pale ribbons, grey-black wide-sleeved robe with gold cloud trim; young, never old.")

# 任何相容 OpenAI images/edits 的端點；在設定檔加 "editUrl" 指定
EDIT_URL = None
MODEL = "gpt-image-2.5"
SIZE = "2048x1152"  # 16:9，邊長為 16 的倍數
_CFG = json.load(open(Path.home() / ".codex" / "niu-image-gen-config.json", encoding="utf-8"))
KEY = _CFG["apiKey"]
EDIT_URL = _CFG.get("editUrl", "https://api.openai.com/v1/images/edits")

STYLE = ("Style: match the style reference CG exactly — the painterly semi-realistic anime illustration of the visual "
         "novel \"哀鴻：城破十日記\": soft visible brush texture, muted desaturated palette, deep soft shadows, "
         "cinematic lighting, same line weight and rendering as the CG. Late Ming dynasty Yangzhou, China.")
FACE = ("Main character: the young Chinese woman in the references. Her face must match the face close-up exactly: "
        "slender oval face, small pointed chin, large slightly drooping almond-shaped warm brown eyes (both eyes the "
        "same brown) with dark upper lashes and faint shadows under the eyes, pale skin, small thin-lipped mouth, quiet "
        "melancholic look. Do not make her generic, older, rounder-faced or doll-like. Outfit: {look}.")


def build(shot, base=None, facefix=False):
    """base：要改的底圖（原版 CG 或已留下的圖）；facefix：只重畫臉，其餘保持不動。"""
    look, cg = PLAN[shot["id"]]
    imgs, roles = [], []
    if base:
        imgs.append(Path(base).resolve()); roles.append("the BASE image to edit")
    if look:
        ref, desc = LOOKS[look]
        who = "his" if look == "fangzhiyou" else "her"
        imgs += [ROOT / "原立绘" / "_臉部" / f"{look}.png", ref]
        roles += [f"{who} face close-up", f"{who} full outfit"]
    if shot.get("him") and look != "fangzhiyou":  # 兩人同框：再附他的立繪
        imgs.append(ROOT / "原立绘" / "方知宥_立繪.png"); roles.append("the young man Fang Zhiyou's full outfit")
    imgs.append(CG / f"{cg}.png"); roles.append("the style reference CG")
    for e in shot.get("refs", []):
        imgs.append(ROOT / e); roles.append("the exact original design of the sachet prop (reproduce it faithfully)")
    if base and facefix:
        task = ("Edit image 1. Keep everything in image 1 exactly as it is — composition, pose, expression, clothing, "
                "background, lighting, colors and brushwork. Only redraw the young woman's face (and eye color) so it "
                "matches her face close-up reference precisely.")
    elif base:
        task = ("Edit image 1 into a new wide 16:9 cinematic still for an anime music video. Keep image 1's painterly "
                "rendering, color palette, lighting mood and overall composition as the starting point, but change the "
                "content to the scene described below.")
    else:
        task = "Create one wide 16:9 cinematic still for an anime music video."
    parts = [task,
             "Reference images in order: " + "; ".join(f"image {i + 1} is {r}" for i, r in enumerate(roles)) + ".",
             STYLE]
    if look == "fangzhiyou":
        parts.append(FACE_M.format(look=desc))
    elif look:
        parts.append(FACE.format(look=desc) + " " + HIM)
    parts += ["Scene: " + shot["p"],
              "Clean finished rendering: no noise, no film grain, no JPEG artifacts, no blotchy texture.",
              "Any visible hands must be anatomically correct: exactly five fingers each, natural knuckles bending the "
              "right way, no fused, extra or twisted fingers.",
              "No text, no letters, no signature, no watermark, no frame, no split panels."]
    return "\n".join(parts), imgs, look, cg


def post(prompt, imgs):
    boundary = "----niu" + str(random.randrange(10**12))
    chunks = []
    def field(name, value):
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    field("model", MODEL); field("prompt", prompt); field("n", "1"); field("size", SIZE)
    for p in imgs:
        chunks.append((f'--{boundary}\r\nContent-Disposition: form-data; name="image[]"; filename="{Path(p).name}"\r\n'
                       "Content-Type: image/png\r\n\r\n").encode() + Path(p).read_bytes() + b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(EDIT_URL, data=b"".join(chunks), headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return base64.b64decode(json.loads(r.read())["data"][0]["b64_json"])


def gen(shot, i, base=None, facefix=False, name_tag=""):
    tag = ("fix-" if facefix else "b-") + Path(base).stem + "_" if base else ""
    tag += f"{name_tag}_" if name_tag else ""
    out = ROOT / "renders" / "niu" / f"{shot['id']}_{tag}{i}.png"
    prompt, imgs, look, cg = build(shot, base, facefix)
    t = time.time()
    for attempt in range(4):  # 503／429 退避重試
        try:
            out.write_bytes(post(prompt, imgs))
            break
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}"
            if e.code not in (429, 500, 502, 503, 504):
                return f"{out.name} 失敗：{err}"
        except Exception as e:  # 逾時等
            err = repr(e)[:300]
        time.sleep(30 * (attempt + 1))
    else:
        return f"{out.name} 失敗：{err}"
    json.dump({"shot": shot["id"], "base": str(base) if base else None, "facefix": facefix, "look": look, "style_cg": cg, "model": MODEL, "size": SIZE,
               "refs": [str(Path(p).resolve().relative_to(ROOT.resolve())) for p in imgs], "prompt": prompt},
              open(out.with_suffix(".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return f"{out.name}  {time.time() - t:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots", nargs="*")
    ap.add_argument("--src", help="依 shots.json 的 src 篩選")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--base", help="底圖路徑（只能配一個鏡號）：以原版 CG 或已選圖為起點改")
    ap.add_argument("--facefix", action="store_true", help="配合 --base：只重畫臉")
    ap.add_argument("--scene", help="覆蓋 shots.json 的場景描述（英文）")
    ap.add_argument("--tag", default="", help="輸出檔名加註（同一底圖做多個版本時用）")
    a = ap.parse_args()
    shots = {s["id"]: s for s in json.load(open(ROOT / "storyboard" / "shots.json", encoding="utf-8"))["shots"]}
    ids = [k for k, s in shots.items() if s.get("src") == a.src and k in PLAN] if a.src else a.shots
    (ROOT / "renders" / "niu").mkdir(parents=True, exist_ok=True)
    tasks = []
    for sid in ids:  # 編號接在現有檔案之後
        start = max([int(p.stem.split("_")[-1]) for p in (ROOT / "renders" / "niu").glob(f"{sid}_*.png")] or [0])
        shot = {**shots[sid], "p": a.scene} if a.scene else shots[sid]
        tasks += [(shot, start + i, a.base, a.facefix, a.tag) for i in range(1, a.n + 1)]
    with ThreadPoolExecutor(a.jobs) as ex:
        for msg in ex.map(lambda t: gen(*t), tasks):
            print(msg, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
