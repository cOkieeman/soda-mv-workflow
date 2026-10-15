# 合成引擎（scripts/）

## 環境

- Python 3.10+：numpy、opencv-python、Pillow、imageio-ffmpeg。
- GPU 版另需 PyTorch CUDA。Windows 上可直接用 ComfyUI 便攜版內嵌的 Python 執行 `compose_gpu.py`（不要往裡面裝東西）。
- 字型：`subs.py` 與 `compose.py` 開頭的 `SERIF`、`KAI` 改成你機器上的字型路徑。
- 本機 ComfyUI：`gen.py` 的 `HOST`、`WF`（API 格式 workflow 路徑）、LoRA 名稱依你的環境改。

## 專案目錄

```
專案/
  compose.py compose_gpu.py subs.py timelines.py   ← 引擎放在專案根目錄
  src/song.mp3
  analysis/lyrics.json      歌詞時間軸（格式見 lyrics_example.json）
  renders/...               定稿圖；renders/hd/<名>_seedvr2.png 會被自動優先使用
  clips/<名>/frame_*.png    動畫逐格
  layers/<名>/depth.png     深度圖（layers.py 產生）
  out/
```

## 指令

```bash
python compose.py s1                         # CPU，24fps，輸出 out/s1.mp4
python compose_gpu.py s1 --fps 60            # GPU，60fps
python compose_gpu.py full --fps 60 --out out/MV_v2.mp4
python compose_gpu.py full --check 128.0     # 只算這一秒，和 CPU 版比對
```

## 鏡頭欄位

| 欄位 | 說明 |
|---|---|
| `t0`, `t1` | 起訖（歌曲秒數） |
| `src` | 單張圖 |
| `seq` | 逐格資料夾，自動依鏡頭長度取樣；`frames: (a, b)` 只取一段；`pingpong` 來回播 |
| `timemap` | 分段變速 `[(鏡頭內秒, 素材秒), ...]`（GPU 版） |
| `resize` | 逐格先縮放到 (w, h) |
| `xfade`, `xdur` | 鏡頭內交叉淡化到其他圖（表情差分） |
| `cam` | `(z0, z1, (cx0, cy0), (cx1, cy1))` 縮放與畫面中心從頭到尾緩動 |
| `ease_in` | 運鏡緩入 |
| `depth`, `par` | 深度圖與視差強度 |
| `fx` | 特效列表，見下 |
| `tin` | 進場轉場，見下 |
| `black` | 整鏡黑畫面（驟停） |

### 轉場 `tin`

| 值 | 效果 | 適合 |
|---|---|---|
| `("dissolve", 秒)` | 溶接 | 通用；退出回憶 |
| `("dim", 秒)` | 邊溶邊微暗 | 回憶淡出再浮現 |
| `("bloom", 秒, x, y, (R,G,B))` | 從 (x, y) 的光暈擴散出下一鏡 | 提燈、從道具泛光進回憶 |
| `("luma", 秒)` | 暗部先溶，像墨暈開 | 墨色畫面；**臉部附近不要用** |
| `("flash", 秒, (R,G,B))` | 白閃切 | 高潮快速閃回 |
| `("dip", 秒)` | 從黑淡入 | 段落開頭 |

### 特效 `fx`

| 值 | 效果 |
|---|---|
| `("flicker", x, y, r, (R,G,B), k)` | 燈火呼吸 |
| `("sweep", "right", (R,G,B), k, t0, t1)` | 燈光從一側掃入 |
| `("rain", n, k)` | 雨絲 |
| `("waterdrops", [(秒, x, y, 半徑)], (開始, 結束) or None)` | 水滴落紙暈開；第二個參數給時間就慢慢暈滿整張 |
| `("sparkle", x, y, 秒)` | 眼淚反光一閃 |
| `("dream", k)` | 夢境暈光柔焦 |

`timemap`、`waterdrops` 只有 GPU 版（compose_gpu.py）支援。

程式畫的花瓣、雨絲疊在手繪畫面上容易像貼紙。能讓畫面本身的元素動（用動畫）就不要疊。

## 段落與完整版

`timelines.py` 裡每段一個時間軸；`build_full()` 依 `ORDER` 串成一條，段落交界不閃黑，前一段最後一鏡自動延長到下一段開始。片尾（outro）另外輸出再用 ffmpeg 串接。

## 字幕與字卡（subs.py）

- 歌詞來自 `analysis/lyrics.json`，`style: "vert"` 直書、`"flow"` 依留白擺放。
- 時間軸 `sub_pos` 可覆蓋某句位置：`{起始秒: {"pos": (x, y), "align": "left"}}` 或 `{"side": "left"}`。
- 片名：`"titles": [(t0, t1, "第一夜", "ヨルシカ", (x, y))]`。放在背景乾淨處（夜空），不要壓在雜亂的頭髮上或擋住人物視線。
- 片尾字卡：outro 時間軸的 `cards`；片尾配樂 `"bgm": (路徑, 起始秒, dB, 淡入秒, 淡出秒)`。
