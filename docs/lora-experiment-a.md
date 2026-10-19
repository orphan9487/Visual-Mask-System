# 三張照片 LoRA：A 組步數對照

來源任務：`5f1e2ab448cd4809adad9cd4ac5a8b64`（李1）。
實驗任務：`990e32dba22f401eac209582b8743556`。

從 SD1.5 底座重新訓練；唯一參數變更是最大 steps 300 → 600。
rank/alpha 8/8、UNet LR 5e-5、batch/accumulation 1/1、seed 42；文字編碼器凍結。
沿用三張照片的順序、同一個 trigger 與同一句 caption，不加入外觀提示、不使用 FaceID。
每 100 steps 保存權重，每一步記錄 loss、照片索引和 AMP scaler。
實驗不加入使用者面具選單、不產生候選基底、不替換正式權重。

訓練：

```cmd
.venv\Scripts\python.exe scripts\evaluation\run_lora_experiment_a.py --source-job 5f1e2ab448cd4809adad9cd4ac5a8b64
```

每次執行建立不同實驗 ID；請使用該次輸出的 ID 執行評估：

```cmd
.venv\Scripts\python.exe scripts\evaluation\evaluate_lora_experiment.py --job 990e32dba22f401eac209582b8743556
```

評估用相同 prompt、LoRA 1.0、seed 42/123/314159、30 inference steps、CFG 7，產生 18 張對照。
InsightFace cosine 與各參考照逐一計算；不等同成功率，也不能單憑分數判斷過擬合。
比較時須一併檢查身分、角度、背景與裁切。CUDA 的數值非確定性可能造成重新訓練的微小差異。
