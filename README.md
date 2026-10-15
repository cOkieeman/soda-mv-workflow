# soda-mv-workflow

用 AI 替遊戲／小說角色配一首歌做 MV 的完整工作流，可裝進 Claude Code 當 skill／插件使用。

```
讀原作 → 分鏡對歌詞 → 定畫風 → 動畫可行性測試 → 生圖審圖 → AI 動畫 → 2.5D 合成＋雙語字幕 → 小樣確認 → 1080p60 出片
```

Claude 負責分鏡、提示詞、審圖、動畫與剪輯；你負責方向、挑圖、給回饋。

第一支實作：《哀鴻：城破十日記》林翩翩 × ヨルシカ〈第一夜〉（4:44，1080p60）。過程中踩過的坑和解法都寫在 [case-study.md](skills/anime-mv-director/references/case-study.md)。

## 包含什麼

| 路徑 | 內容 |
|---|---|
| `skills/anime-mv-director/SKILL.md` | 流程、關卡與鐵則（Claude 讀這份） |
| `references/workflow.md` | 各階段細節 |
| `references/review-checklist.md` | 交圖前檢查：臉、瞳色、手、道具、構圖 |
| `references/animation.md` | 動畫幀率問題與解法、即夢／MiniMax H3 操作 |
| `references/compose-engine.md` | 合成引擎用法、鏡頭欄位、轉場與特效 |
| `references/copyright.md` | 版權與素材原則 |
| `references/cinematic-director.md` | 搭配 [cinematic-director](https://github.com/cOkieeman/xiaoyou-aigc-cinematic-director) 產生攝影級提示詞 |
| `scripts/` | 合成引擎（CPU／GPU）、字幕引擎、本機 ComfyUI 生圖／去噪／圖轉影片、時間軸範本 |

不包含任何原作素材、歌曲或歌詞。

## 安裝

Claude Code 插件：

```bash
claude plugin marketplace add cOkieeman/soda-mv-workflow
```

```bash
claude plugin install soda-mv-workflow@soda-mv-workflow
```

或只用 skill：把 `skills/anime-mv-director/` 複製到 `~/.claude/skills/`。

## 需要的環境

- Python 3.10+：numpy、opencv-python、Pillow、imageio-ffmpeg；GPU 合成另需 PyTorch CUDA。
- 本機生圖／去噪／H3 動畫：ComfyUI（API 格式 workflow 自備，路徑與 LoRA 名稱在 `scripts/gen.py` 開頭改）。
- 動畫：即夢 Seedance（網頁）或本機 MiniMax H3。
- 字型：日文襯線體、中文楷體（在 `scripts/subs.py` 開頭改路徑）。

腳本是從第一支實作直接整理出來的，路徑與模型名稱依你的環境修改；`scripts/examples/` 是當時的角色專用生圖腳本，僅供參考。

## 授權

MIT。
