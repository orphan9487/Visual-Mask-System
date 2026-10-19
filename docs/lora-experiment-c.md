# C 組：降低學習率

來源面具：李1，`5f1e2ab448cd4809adad9cd4ac5a8b64`。
C 組任務：`135d259b12264c7c83bb7b325ceedc71`。

從 SD1.5 底座重新訓練；固定 rank/alpha 8/8、500 steps、batch/accumulation 1/1、seed 42。
照片順序、trigger、caption、凍結文字編碼器均沿用來源。
唯一設定差異為 UNet LR 5e-5 → 2e-5。

對照 A 組 `990e32dba22f401eac209582b8743556` 的 500-step 權重。
相同中性 prompt、LoRA 1.0、CFG 7、30 inference steps、seed 42/123/314159。
不加入 FaceID 或外觀描述，不替換正式面具。

```cmd
.venv\Scripts\python.exe scripts\evaluation\run_lora_experiment_a.py --source-job 5f1e2ab448cd4809adad9cd4ac5a8b64 --rank 8 --steps 500 --learning-rate 2e-5
.venv\Scripts\python.exe scripts\evaluation\evaluate_lora_experiment.py --job 135d259b12264c7c83bb7b325ceedc71 --compare-job 990e32dba22f401eac209582b8743556
```

較低學習率在相同步數下可能學得較慢；結果只比較固定預算下的表現，不能單獨證明最佳收斂品質。
