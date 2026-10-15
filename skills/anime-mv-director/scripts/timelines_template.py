"""MV 時間軸範本：複製成 timelines.py 放在專案根目錄，再依分鏡改。

時間都是歌曲秒數。欄位說明見 references/compose-engine.md。
"""
from pathlib import Path

ROOT = Path(__file__).parent
AMBER = (255, 176, 92)    # 燈籠暖光
CANDLE = (255, 150, 70)   # 燭光
SONG = "src/song.mp3"


def seq(folder):
    """逐格資料夾：有 SeedVR2 放大版 hd/ 就用它。"""
    hd = ROOT / folder / "hd"
    return f"{folder}/hd" if hd.exists() and any(hd.glob("frame_*.png")) else folder


def depth(name):
    p = ROOT / "layers" / name / "depth.png"
    return f"layers/{name}/depth.png" if p.exists() else None


TIMELINES = {
    # 片頭：靜態圖推鏡＋燈火，片名放在背景乾淨處
    "s0": {
        "t0": 0.0, "t1": 20.0, "audio": SONG, "fade_in": 1.5, "fade_out": 0.3,
        "shots": [
            {"t0": 0.0, "t1": 8.0, "src": "renders/01.png", "ease_in": True,
             "cam": (1.0, 1.15, (0.5, 0.5), (0.55, 0.45)),
             "fx": [("flicker", 0.7, 0.6, 0.2, AMBER, 0.3)]},
            # 從燈光擴散進下一鏡
            {"t0": 8.0, "t1": 20.0, "src": "renders/02.png", "tin": ("bloom", 1.8, 0.62, 0.45, AMBER),
             "cam": (1.3, 1.1, (0.3, 0.6), (0.6, 0.6))},
        ],
        "dust": (45, (255, 210, 160)),     # 浮塵粒子
        "titles": [(16.0, 20.0, "曲名", "歌手", (1660, 120))],
        "sub_pos": {8.0: {"side": "left"}},
    },
    # 主歌：動畫鏡頭（即夢 24fps 慢動作生成，鋪滿鏡頭長度＝60fps 播放）
    "a1": {
        "t0": 20.0, "t1": 40.0, "audio": SONG,
        "shots": [
            {"t0": 20.0, "t1": 31.8, "seq": "clips/jm_01", "resize": (1920, 1080),
             "tin": ("dissolve", 1.0),
             # 小動作慢放、轉場快放：(鏡頭內秒, 素材秒)
             "timemap": [(0.0, 4.0), (5.0, 10.5), (7.5, 18.0), (11.8, 30.0)]},
            # 回憶：從前一鏡的道具位置泛白光進場
            {"t0": 31.8, "t1": 35.0, "src": "renders/memory.png", "tin": ("bloom", 1.4, 0.47, 0.37, (255, 250, 240))},
            # 退出回憶：柔和溶接
            {"t0": 35.0, "t1": 40.0, "src": "renders/03.png", "tin": ("dissolve", 1.2),
             "fx": [("rain", 200, 0.6)]},
        ],
    },
    # 片尾字卡（無畫面，配另一首曲子）
    "outro": {
        "t0": 0.0, "t1": 22.0, "audio": None, "fade_in": 0.01, "fade_out": 1.5,
        "bgm": ("src/outro.mp3", 1.0, 4.0, 2.0, 3.0),   # (路徑, 起始秒, dB, 淡入, 淡出)
        "shots": [],
        "cards": [
            (2.0, 8.0, [("獻給 某某", "big", 2.0)]),
            (9.0, 16.0, [("原作　《作品名》", "zh", 1.7), ("歌曲　歌手〈曲名〉", "zh", 1.7),
                         ("", "small", 0.8), ("製作　你的名字", "zh", 1.7)]),
        ],
    },
}

ORDER = ["s0", "a1"]


def build_full():
    """把各段串成一條時間軸，段落交界不閃黑。"""
    import copy
    shots, sub_pos, titles = [], {}, []
    for k, name in enumerate(ORDER):
        tl = copy.deepcopy(TIMELINES[name])
        seg = tl["shots"]
        for s in seg:
            s.setdefault("dust_cfg", tl.get("dust"))
            s.setdefault("grain_cfg", tl.get("grain"))
        if k > 0 and shots:
            shots[-1]["t1"] = seg[0]["t0"]
        shots += seg
        sub_pos.update(tl.get("sub_pos", {}))
        titles += tl.get("titles", [])
    last = TIMELINES[ORDER[-1]]
    return {"t0": TIMELINES[ORDER[0]]["t0"], "t1": last["t1"], "audio": SONG, "fade_in": 2.0, "fade_out": 2.0,
            "shots": shots, "sub_pos": sub_pos, "titles": titles}


TIMELINES["full"] = build_full()
