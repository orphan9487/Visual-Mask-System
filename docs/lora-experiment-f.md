# F 組：背景統一

任務：`dd544bb97dbf4c85bf0d8a5b32e0104d`。
基準：E 組 `09a4fa25e30842b69928f8a0865876fd`。

使用 E 組相同的三張 512px 圖片，只以 U2-Net 遮罩替換背景為 RGB(127,127,127)。
不改構圖、人物像素、caption 或訓練參數；柔化邊緣由遮罩決定，輪廓可能有少許殘留。
rank/alpha 8/8、LR 5e-5、500 steps、seed 42、batch 1、文字編碼器凍結。
先生成三張原圖／遮罩／合成圖對照，目視確認後才另行啟動訓練。

去背模型下載自 rembg release 的 u2net.onnx，核對 MD5 `60024c5c889badc19c04ad937298a77b`。
本機 ONNX Runtime CPU 推論，前後處理依 rembg U2netSession（320px、ImageNet normalization、min-max mask）。
實驗準備期間曾下載 rembg、pymatting、pooch、platformdirs 至 `tmp/background_experiment_deps`；因欠缺 numba，最終改用既有 ONNX Runtime，不需要匯入這些額外套件，現有虛擬環境套件未變更。
模型放在 `models/background_removal/u2net.onnx`，照片不離開本機。

```cmd
.venv\Scripts\python.exe scripts\evaluation\run_lora_background_experiment.py --source-job 09a4fa25e30842b69928f8a0865876fd
.venv\Scripts\python.exe scripts\evaluation\run_lora_background_experiment.py --train-job dd544bb97dbf4c85bf0d8a5b32e0104d
.venv\Scripts\python.exe scripts\evaluation\evaluate_lora_experiment.py --job dd544bb97dbf4c85bf0d8a5b32e0104d --compare-job 09a4fa25e30842b69928f8a0865876fd
```

評估固定相同中性 prompt、三組生成 seed、LoRA 1.0；兩組相似度均對最初三張 processed 參考照計算，非去背後照片。
不註冊正式面具、不修改正式訓練預設與聊天室。

## 結果

500-step 訓練、六張生成對照與相似度評估完成；兩支腳本語法檢查通過。

| seed | E 保留背景 | F 統一灰底 |
| --- | ---: | ---: |
| 42 | 0.4055 | 0.3813 |
| 123 | 0.3685 | 0.4397 |
| 314159 | 0.3288 | 0.4828 |
| 平均 | 0.3676 | 0.4346 |

平均增加約 0.0670 cosine，兩個 seed 改善、一個下降。
目視 F 三張均保留短髮與眼鏡，E 第三張的眼鏡消失／髮型偏移在此輪未出現。
F 第一張仍生成棕色背景，人物角度也不完全一致，因此不能宣稱背景／角度已完全可控。
這是單次訓練與三個生成 seed，對訓練照片評分，不是獨立測試準確率；尚未驗證 hybrid 七種表情。
暫不修改正式預設，建議下一步用此 checkpoint 做 hybrid 端到端驗證。

預覽：`output/current/lora_background_experiment/dd544bb97dbf4c85bf0d8a5b32e0104d/preprocessing_comparison.png`。
生成對照與 report：`output/current/lora_rank_comparison/dd544bb97dbf4c85bf0d8a5b32e0104d/`。
