# 搭配 cinematic-director

[cOkieeman/xiaoyou-aigc-cinematic-director](https://github.com/cOkieeman/xiaoyou-aigc-cinematic-director) 是同一作者的「視覺導演」skill：把鏡頭描述變成電影攝影級生圖提示詞（攝影語言、光影、構圖、色板、正負向 prompt），用模擬攝影團隊開會的方式決策。

兩個倉庫分開維護，本 skill 不複製它的檔案，只說明怎麼接。

## 放在哪一步

```
1 概念與分鏡（shots.json：每鏡畫面、情緒、時間）
        ↓
   cinematic-director：逐鏡產出攝影方案與正負向 prompt
        ↓
2 定畫風 / 4 生圖（把 prompt 套進本機模型或 GPT 生圖，加角色 LoRA 觸發詞與參考圖）
```

## 怎麼用

1. 照它的 README 安裝（把 `cinematic-director/` 資料夾放進 Claude Code 的 skills）。
2. 每鏡給它：場景、角色、情緒、這句歌詞、前後鏡頭、基準畫風（例如「原作 CG：柔和筆觸、低飽和、深陰影」）。
3. 拿回的攝影方案重點用在：景別、鏡頭高度、主光方向與色溫、前景遮擋、景深。
4. 生圖時仍要加上本專案的固定部分：角色 LoRA 觸發詞、瞳色、造型描述、負面詞（手、多餘手指、文字、其他瞳色）。

## 注意

- 它的提示詞偏寫實電影感；動漫插畫 MV 要再加上基準畫風描述，否則會和原 CG 脫節。
- 「避免擺拍」的原則要另外告訴它：要抓拍感，人物不要對鏡頭展示道具。
- 同系列的編劇（story-architect）、分鏡（libtv-storyboard）skill 另行提供；沒有它們時，本流程的分鏡由 Claude 依歌詞與原作完成。
