# E 組：統一人臉構圖

任務：`09a4fa25e30842b69928f8a0865876fd`。
基準：A 組 `990e32dba22f401eac209582b8743556` 的 500-step checkpoint。

使用同三張已處理照片，透過 InsightFace 單人臉框建立正方形，邊長為人臉框長邊的 1.5 倍，中心向上移動臉高的 0.08 倍，等比例縮放至 512px。
超出來源影像的區域以灰色補齊，不拉伸、不翻轉、不去背；保留原檔。
此實驗驗證構圖一致性，並非人像分割或完整背景移除。原本已裁切掉的內容無法恢復；補灰也可能形成新的偏差。

固定 rank/alpha 8/8、UNet LR 5e-5、500 steps、batch 1、seed 42，文字編碼器不訓練，trigger/caption 不變。
無資料庫寫入、不註冊面具、不修改正式訓練預設。

```cmd
.venv\Scripts\python.exe scripts\evaluation\run_lora_crop_experiment.py --source-job 990e32dba22f401eac209582b8743556
.venv\Scripts\python.exe scripts\evaluation\evaluate_lora_experiment.py --job 09a4fa25e30842b69928f8a0865876fd --compare-job 990e32dba22f401eac209582b8743556
```

評估使用相同中性人像 prompt、seed 42/123/314159、LoRA 1.0。
兩組皆對原始三張參考照計算 cosine，避免更換評分參考而產生偏差。
評估器核對來源路徑及 SHA256，允許經稽核的構圖變更，其他不相同輸入仍拒絕比較。

## 結果

| seed | 原本前處理 A | 統一構圖 E |
| --- | ---: | ---: |
| 42 | 0.2852 | 0.4055 |
| 123 | 0.2928 | 0.3685 |
| 314159 | 0.4675 | 0.3288 |
| 平均 | 0.3485 | 0.3676 |

三個 seed 兩個提升、一個下降，平均僅增加約 0.0191 cosine，不足以宣稱穩定改善。
E 的前兩張較接近預期短髮人像，但第三張眼鏡未保留、髮型仍偏移；第一張也未遵守純色背景。
因此尚未套用正式預設。構圖調整未解決全部身分／背景糾纏問題。
這是單次訓練、三個生成 seed，且分數對訓練參考照計算，不是獨立驗證集或準確率。

輸入對照：`output/current/lora_crop_experiment/09a4fa25e30842b69928f8a0865876fd/preprocessing_comparison.png`。
生成對照與 report：`output/current/lora_rank_comparison/09a4fa25e30842b69928f8a0865876fd/`。
裁切幾何單元測試 2 項通過；500-step 訓練與六張對照生成完成。
