# LoRA 載入與人物重現診斷

執行命令：

```cmd
.venv\Scripts\python.exe scripts\evaluation\diagnose_lora_loading.py --job 990e32dba22f401eac209582b8743556
```

固定 30 inference steps、CFG 7、512px、seed 42/123；無 FaceID、無負面提示詞。
兩種 prompt：訓練時原始 caption，以及額外要求正面攝影棚人像的 caption。
共 28 張：不載入 LoRA，加上 300/500 steps 各以 0.5/1.0/1.5 權重生成。

## 結果

- 三個輸入檔均能偵測單張人臉。
- 300 與 500 checkpoint SHA256 不同；每份 256 個非空、有限數值張量，載入後 UNet 有 256 個 LoRA 參數張量，diagnostic adapter 已啟用。
- 固定 seed 的生成圖隨權重改變，排除本次獨立診斷中「LoRA 完全沒有作用」的情況。
- 訓練 caption、500 steps、權重 1.0：seed 42/123 對三張訓練參考照平均 cosine 分別 0.4198/0.4015。相同 seed 的不載入版本為多人拼貼（不評分）及 -0.0377。
- 改為 studio caption 後，同設定分數為 0.2706/0.1928，顯示人物重現對生成條件敏感。
- 0.5 權重仍明顯受到基底模型人物外觀影響；1.5 多張觸發安全檢查，無法當作改善證據。黑圖不參與相似度評估，安全檢查未停用。
- 目視 500/1.0 訓練 caption 生成更接近訓練人物，但背景也跟著學入；不能認為已達穩定可用。
- `src/services/mask_bases.py` 正式候選生成仍硬編碼 300 steps 與 0.7 權重，與實驗較佳條件不同。本輪未更動正式流程，也未以本輪實驗冒充正式模型結果。

## 判讀限制

本測試確認的是獨立 SD1.5 文字生圖載入鏈路，不代表聊天室 hybrid 的端到端載入已全部通過。
分數是訓練參考照的 cosine，不是百分比或獨立測試準確率；只有兩個生成 seed。
無法據此判定 LoRA 普遍不適用；目前證據較符合「已學到部分身分，但與背景糾纏，且生成條件敏感」。
完整 report.json、圖片與兩張對照圖在 `output/current/lora_loading_diagnostic/990e32dba22f401eac209582b8743556/`。
