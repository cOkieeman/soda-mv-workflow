"""第一夜 MV：透過本機 ComfyUI API 依 shots.json 生圖。

用法：
  python gen.py 13 52 --n 4                       # 指定鏡頭，各出 4 張（模型依 shots.json 的 m）
  python gen.py 13 --model A --style star         # 改用 Anima＋star cat 畫風
  python gen.py 52 --i2i renders/codex/52_2.png --denoise 0.4 --model A   # 圖生圖統一畫風／修臉
  python gen.py 13 --look liuxiang --cw 0.8       # 指定造型與角色 LoRA 權重
  python gen.py 13 --hires                        # 保留 SeedVR2 超分（定稿用）
輸出：renders/<鏡號>_<模型><style>_<seed>.png（圖生圖為 <來源檔名>_<模型><style>_d<denoise>_<seed>.png），
同名 .json 記錄 prompt 與參數。
"""
import argparse, copy, json, random, sys, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
HOST = "http://127.0.0.1:8190"
WF = {
    "K": ROOT / "workfl" / "Soda-Krea2-文生图-v2-N卡超分.json",
    "K_i2i": ROOT / "workfl" / "Soda-Krea2-图生图-v2-N卡超分.json",
    "A": ROOT / "workfl" / "Soda-Anima-v5-N卡超分.json",
}

EYES = "brown eyes, both eyes the same warm brown color"
# 造型 → 各模型的角色 LoRA（完整路徑，Anima 有兩個 V1.safetensors）與觸發詞
LOOKS = {
    "liuxiang_eye": {
        "K": ("Krea2\\character\\linhuier-epoch-08.safetensors",
              "linhuier, Her long side-swept bangs completely cover one eye, leaving the other eye visible. "
              f"{EYES}, small side bun, very long dark brown hair, plain grey-green hanfu robe, pale long skirt."),
        "A": ("Anima\\character\\linhuier-anima_v013-train118-reg118-r64-30ep_epoch26.safetensors",
              f"1girl, linhuier, solo, hair over one eye, {EYES}, very long dark brown hair, small side bun, "
              "grey-green hanfu, pale skirt, long sleeves,"),
    },
    "liuxiang": {
        "K": ("Krea2\\character\\linhuier-epoch-08.safetensors",
              f"linhuier, Her bangs are parted so both eyes are fully visible. {EYES}, small white flower hairpin, "
              "very long dark brown hair, plain grey-green hanfu robe, pale long skirt."),
        "A": ("Anima\\character\\linhuier-anima_v013-train118-reg118-r64-30ep_epoch26.safetensors",
              f"1girl, linhuier, solo, {EYES}, very long dark brown hair, flower hair ornament, "
              "grey-green hanfu, pale skirt, long sleeves,"),
    },
    "huajie": {
        "K": ("Krea2\\character\\linhuier-epoch-08.safetensors",
              f"linhuier, Her bangs are parted so both eyes are fully visible. {EYES}, gold hair ornament with tassels, "
              "pearl necklace, dark green outer robe with white trim, white long skirt."),
        "A": ("Anima\\character\\linhuier-anima_v013-train118-reg118-r64-30ep_epoch26.safetensors",
              f"1girl, linhuier, solo, {EYES}, blunt bangs, gold hair ornament, necklace, dark green robe, "
              "white skirt, long sleeves,"),
    },
    "toupai": {
        "K": ("Krea2\\character\\LinPianPianV4.safetensors",
              f"Linpianpian, hair ornament, necklace, green dress, hanfu, {EYES}."),
        "A": ("Anima\\character\\V1.safetensors",
              f"1girl, linpianpian, hair ornament, necklace, hair bun, {EYES}, hair stick, green dress, hanfu,"),
    },
}
DEFAULT_LOOK = {"K": "liuxiang_eye", "A": "toupai"}  # shots.json 的 m=K / m=A

# 畫風 LoRA 組合（LoraManager 名稱 → 權重），用 --style 切換
STYLES = {
    "K": {
        "none": {},
        "base": {"light_c1-st2000": 0.2, "MW_analogv1.1": 0.32, "Krea2PlaneStyle_c1-st8000": 0.3},
        # 使用者的平塗抓拍風（封面用，來自使用者 workflow 2026-09-17-123909）
        "candid": {"MW_analogv1.1": 0.40, "Krea2Rella_c1-st8000": 0.19, "betabeet-k2-4_c1-st6000": 0.13,
                   "ANIRL-KREA2_epoch_4": 0.20},
    },
    "A": {
        "none": {},
        "base": {"anima-highres-aesthetic-boost": 0.2, "anima-base-1-masterpiece-v51": 0.2},
        "star": {"star cat-000022": 0.8, "anima-highres-aesthetic-boost": 0.4, "anima-base-1-masterpiece-v51": 0.4},
        # 使用者的平塗畫風（workflow oneObsessionAnima_v30_2026-10-08-202941），底模換 v30、觸發詞 @a1g2m3 @m4me
        "obs": {"Ymstyle_animaV1.0": 0.70, "anima_m4me_v2.1-epoch20": 0.60, "anima-highres-aesthetic-boost": 0.35,
                "agm-000027": 0.35, "mature body": 0.20},
    },
}
DEFAULT_STYLE = {"K": "base", "A": "star"}

BAD_QUALITY = "noise, film grain, jpeg artifacts, blotchy, bad hands, extra fingers, fused fingers, missing fingers, malformed knuckles"
BAD_EYES = "red eyes, blue eyes, green eyes, grey eyes, purple eyes, yellow eyes, heterochromia, mismatched eye colors"
NEG = {
    "K": "text, watermark, logo, signature, letters, modern clothing, modern building, photo, 3d render, "
         "bad anatomy, extra fingers, malformed hands, blurry face, low detail face, distorted eyes, "
         f"duplicate person, nsfw, nude, cleavage, {BAD_EYES}, {BAD_QUALITY}",
    "A": "worst quality, low quality, score_1, score_2, score_3, artist name, old, normal quality, "
         "anatomical nonsense, bad anatomy, interlocked fingers, extra fingers, watermark, logo, text, "
         f"signature, nsfw, nude, cleavage, {BAD_EYES}, {BAD_QUALITY}",
}


CANDID_PREFIX = ("Photography Style: soft candid summer portrait, airy natural light, slight bloom, shallow depth of "
                 "field, soft foreground haze, soft background blur, understated emotional warmth, quiet "
                 "slice-of-life mood. Subject:")
CANDID_NEG = ("worst quality, low quality, blurry face, bad anatomy, bad hands, extra fingers, fused fingers, "
              "missing fingers, toothy grin, open-mouth smile, exaggerated smile, fake smile, fashion photoshoot pose, "
              "idol pose, promotional pose, dramatic pose, holding props toward the camera, sharp everything, "
              "no depth of field, hard poster composition, heavy fantasy glow, overdone bloom, multiple girls, "
              "duplicate character, 3d render, glossy anime rendering, watermark, logo, text, signature, "
              f"modern clothing, {BAD_EYES}")


def api(path, data=None):
    req = urllib.request.Request(HOST + path, data=json.dumps(data).encode() if data else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def upload(path):
    """把圖片上傳到 ComfyUI input，回傳檔名。"""
    boundary = "----mvdaiichiya" + str(random.randrange(10**9))
    data = Path(path).read_bytes()
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="mv_{Path(path).name}"\r\n'
            'Content-Type: image/png\r\n\r\n').encode()
    tail = f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n--{boundary}--\r\n'.encode()
    req = urllib.request.Request(HOST + "/upload/image", data=head + data + tail,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["name"]


def build(shot, models, fam, style_name, look, cw, seed, hires, i2i=None):
    """fam: K / A；look: 造型名或 None（空景）；i2i: {"name", "denoise"} 或 None。"""
    wf = copy.deepcopy(json.load(open(WF["K_i2i" if i2i and fam == "K" else fam], encoding="utf-8")))
    w, h = map(int, shot["size"].split("x"))

    lora, trig = LOOKS[look][fam] if look else (None, "")
    body = (trig + "\n" if trig else "") + shot["p"]
    for k in ("wildcard_text", "populated_text"):
        wf["184"]["inputs"][k] = body
    wf["184"]["inputs"]["mode"] = "fixed"
    neg = NEG[fam]
    if shot.get("allow_text"):  # 封面等需要畫出文字的圖：負面詞不擋文字
        for w_ in ("text, ", "letters, ", "logo, ", "watermark, ", "signature, "):
            neg = neg.replace(w_, "")
    wf["75"]["inputs"]["string"] = neg
    wf["84"]["inputs"]["seed"] = seed

    candid = fam == "K" and style_name == "candid"
    if candid:
        body = body + ", in ANIRL Style"
        for k in ("wildcard_text", "populated_text"):
            wf["184"]["inputs"][k] = body
        neg = CANDID_NEG
        if shot.get("allow_text"):
            for w_ in ("watermark, ", "logo, ", "text, ", "signature, "):
                neg = neg.replace(w_, "")
        wf["75"]["inputs"]["string"] = neg

    if shot.get("neg_extra"):
        wf["75"]["inputs"]["string"] += ", " + shot["neg_extra"]

    if fam == "K":
        wf["182"]["inputs"]["string"] = CANDID_PREFIX if candid else models["style"]["K"]
        wf["220"]["inputs"]["string"] = ""          # 拿掉 soda 簽名
        wf["143"]["inputs"]["text_a"] = ""          # 觸發詞自己寫，不吃 LoRA 自帶的（有互相矛盾的句子）
        stacks = ("365", "366", "367")
        switch = "152:267"
        if candid:  # 使用者的設定用 beta 排程
            wf["148:146"]["inputs"]["scheduler"] = "beta"
        if i2i:
            wf["315"]["inputs"]["image"] = i2i["name"]
            wf["325"]["inputs"]["float"] = i2i["denoise"]
            wf["370:320"]["inputs"]["megapixels"] = 1.4  # 保持接近 Codex 原尺寸
        else:
            wf["150"]["inputs"]["resolution_json"] = json.dumps({"version": 1, "width": w, "height": h})
    else:
        wf["182"]["inputs"]["string"] = wf["182"]["inputs"]["string"] + " " + models["style"]["A"] + ","
        wf["183"]["inputs"]["string"] = ""
        if style_name == "obs":
            wf["135:111"]["inputs"]["unet_name"] = r"Anima\OneObsession\oneObsessionAnima_v30.safetensors"
            body = body + ", @a1g2m3, @m4me, mature body"
            for k in ("wildcard_text", "populated_text"):
                wf["184"]["inputs"][k] = body
        wf["143"]["inputs"]["text_c"] = ""
        # WidgetToString 要讀 UI workflow，API 送出時沒有，改成直接填值
        wf["81"]["inputs"]["sampler_name"] = "er_sde"
        wf["81"]["inputs"]["scheduler_name"] = "sgm_uniform"
        wf["152:341"]["inputs"]["values.a"] = "1920"
        for n in ("361:351", "361:352", "152:340"):
            wf.pop(n)
        stacks = ("301", "302", "303")
        switch = "152:225"
        wf["150"]["inputs"]["resolution_json"] = json.dumps({"version": 1, "width": w, "height": h})
        if i2i:  # Anima workflow 只有文生圖，自己接 LoadImage → 縮放 → VAEEncode
            wf["950"] = {"class_type": "LoadImage", "inputs": {"image": i2i["name"]}}
            wf["951"] = {"class_type": "ImageScaleToTotalPixels",
                         "inputs": {"upscale_method": "lanczos", "megapixels": 1.4, "resolution_steps": 16,
                                    "image": ["950", 0]}}
            wf["952"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["951", 0], "vae": ["135:65", 0]}}
            wf["148:146"]["inputs"]["latent_image"] = ["952", 0]
            wf["148:146"]["inputs"]["denoise"] = i2i["denoise"]

    # 清空 LoraManager 堆疊，第二個堆疊放畫風
    style = STYLES[fam][style_name]
    for sid in stacks:
        wf[sid]["inputs"]["loras"] = {"__value__": []}
        wf[sid]["inputs"]["text"] = ""
    wf[stacks[1]]["inputs"]["loras"] = {"__value__": [
        {"name": n, "strength": s, "clipStrength": s, "active": True} for n, s in style.items()]}
    wf[stacks[1]]["inputs"]["text"] = " ".join(f"<lora:{n}:{s}>" for n, s in style.items())

    # 角色 LoRA 接在 77 之後
    if lora:
        wf["900"] = {"class_type": "LoraLoader", "_meta": {"title": "MV 角色 LoRA"},
                     "inputs": {"lora_name": lora, "strength_model": cw, "strength_clip": cw,
                                "model": ["77", 0], "clip": ["77", 1]}}
        wf["184"]["inputs"]["model"] = ["900", 0]
        wf["184"]["inputs"]["clip"] = ["900", 1]

    if not hires:  # 拿掉 SeedVR2，直接用 base 圖
        wf[switch]["inputs"].pop("any_01", None)
    return wf, body


def run(wf):
    pid = json.loads(api("/prompt", {"prompt": wf, "client_id": "mv_daiichiya", "front": True}))["prompt_id"]
    while True:
        time.sleep(2)
        hist = json.loads(api(f"/history/{pid}"))
        if pid in hist:
            h = hist[pid]
            st = h.get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(json.dumps(st.get("messages", [])[-1:], ensure_ascii=False)[:2000])
            return h["outputs"]


def fetch(outputs, node):
    for img in outputs.get(node, {}).get("images", []):
        q = urllib.parse.urlencode({k: img[k] for k in ("filename", "subfolder", "type")})
        return urllib.request.urlopen(f"{HOST}/view?{q}", timeout=60).read()
    raise RuntimeError(f"節點 {node} 沒有輸出圖片：{list(outputs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots", nargs="+")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--model", choices=["K", "A"], help="預設依 shots.json 的 m")
    ap.add_argument("--style", help="逗號分隔可做對照；預設 K=base、A=star")
    ap.add_argument("--look", help="造型：" + "/".join(LOOKS) + "；none＝不掛角色 LoRA")
    ap.add_argument("--cw", type=float, default=0.9, help="角色 LoRA 權重（建議 0.8–1.0）")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--hires", action="store_true")
    ap.add_argument("--out", default="renders")
    ap.add_argument("--i2i", help="圖生圖：輸入圖路徑（shots 只能給一個鏡號）")
    ap.add_argument("--denoise", type=float, default=0.4)
    ap.add_argument("--scene", help="覆蓋分鏡的畫面描述（英文）")
    ap.add_argument("--size", help="覆蓋尺寸，例如 1344x768")
    ap.add_argument("--allow-text", action="store_true", help="允許畫面出現文字（封面）")
    ap.add_argument("--neg", default="", help="額外負面詞（接在原負面詞後）")
    a = ap.parse_args()

    models = json.load(open(ROOT / "storyboard" / "shots.json", encoding="utf-8"))
    by_id = {s["id"]: s for s in models["shots"]}
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for sid in a.shots:
        shot = dict(by_id[sid]) if sid in by_id else {"id": sid, "m": a.model or "A", "size": "1344x768", "p": ""}
        if a.scene:
            shot["p"] = a.scene
        if a.size:
            shot["size"] = a.size
        if a.allow_text:
            shot["allow_text"] = True
        if a.neg:
            shot["neg_extra"] = a.neg
        fam = a.model or shot["m"][0]
        # 造型：--look ＞ Codex 圖的 json ＞ shots.json 的 m（K0/A0 為空景）
        look = a.look
        i2i = None
        if a.i2i:
            meta_path = Path(a.i2i).with_suffix(".json")
            meta = json.load(open(meta_path, encoding="utf-8")) if meta_path.exists() else {}
            look = look or meta.get("look")
            i2i = {"name": upload(a.i2i), "denoise": a.denoise}
        if look is None:
            look = DEFAULT_LOOK.get(shot["m"])
        if look == "none":
            look = None
        for i in range(a.n):
            seed = a.seed + i if a.seed is not None else random.randrange(1, 2**40)
            for style in (a.style or DEFAULT_STYLE[fam]).split(","):
                wf, body = build(shot, models, fam, style, look, a.cw, seed, a.hires, i2i)
                t = time.time()
                outputs = run(wf)
                saver = "204" if fam == "K" else "81"
                tag = f"{fam}{style}"
                name = f"{sid}_{Path(a.i2i).stem}_{tag}_d{a.denoise}_{seed}" if i2i else f"{sid}_{tag}_{seed}"
                (out / f"{name}.png").write_bytes(fetch(outputs, saver))
                rating = outputs.get("348", {}).get("text", [""])[0] if fam == "K" else ""
                json.dump({"shot": sid, "model": fam, "style": style, "look": look, "cw": a.cw, "seed": seed,
                           "i2i": a.i2i, "denoise": a.denoise if i2i else None, "prompt": body,
                           "rating": rating, "secs": round(time.time() - t, 1)},
                          open(out / f"{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                print(f"{name}.png  {time.time() - t:.0f}s  {rating}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
