# D 組：文字編碼器 LoRA

來源：李1 `5f1e2ab448cd4809adad9cd4ac5a8b64`。
任務：`4613f006d62c4b128b169d70d6ab9add`。

沿用 A 組三張照片、相同 trigger 與 caption。
UNet rank/alpha 8/8，LR 5e-5；500 steps、batch 1、seed 42。
加入 CLIP text encoder attention q/k/v/out LoRA，rank/alpha 8/8，LR 5e-6。
只更新 LoRA 矩陣，原始文字編碼器與詞嵌入仍凍結；每一步重新計算文字特徵。
每 100 steps 同時保存 UNet 與 text_encoder 權重。
新增 adapter 初始化會改變後續 RNG 序列；此輪是探索性對照，不是多個訓練 seed 的統計實驗。

```cmd
.venv\Scripts\python.exe scripts\evaluation\run_lora_experiment_a.py --source-job 5f1e2ab448cd4809adad9cd4ac5a8b64 --rank 8 --steps 500 --text-encoder-learning-rate 5e-6
.venv\Scripts\python.exe scripts\evaluation\evaluate_lora_experiment.py --job 4613f006d62c4b128b169d70d6ab9add --compare-job 990e32dba22f401eac209582b8743556
```

固定相同中性 prompt、LoRA 1.0、三個生成 seed，無 FaceID，不新增外觀提示。
實驗不加入面具選單，也不取代正式面具設定。

## 結果

500 steps 完成；100-step checkpoint 含 96 個 text_encoder LoRA 張量。
相同生成 seed 的 InsightFace 相似度（對三張訓練照片的 cosine 平均）：

| seed | A：凍結文字編碼器 | D：文字編碼器 LoRA |
| --- | ---: | ---: |
| 42 | 0.2852 | 0.2976 |
| 123 | 0.2928 | 0.1798 |
| 314159 | 0.4675 | 0.3633 |
| 平均 | 0.3485 | 0.2803 |

D 沒有穩定改善身分相似度；seed 123 眼鏡未保留，seed 42 仍有長髮。
不採用為正式預設。這不是準確率，也不是獨立測試集評估；只有單次訓練、三個生成 seed。
對照圖與完整數值位於 `output/current/lora_rank_comparison/4613f006d62c4b128b169d70d6ab9add/`。
