# LoRA rank 容量對照（B 組）

A 組任務 `990e32dba22f401eac209582b8743556` 的 500-step checkpoint
對照 B 組任務 `27dd93a09bd846c1a08762f95941805e` 的 500-step checkpoint。

B 組從同一 SD1.5 底座重新訓練，rank/alpha 由 8/8 → 16/16，比例保持 1。
相同三張照片、順序、trigger、caption、UNet LR 5e-5、batch 1、seed 42；凍結文字編碼器。
rank 改變初始化張量大小，也會改變後續 RNG 序列；固定 seed 不代表兩次訓練每一步噪聲相同。
此為小規模探索，尚非多個訓練 seed 的統計實驗。

```cmd
.venv\Scripts\python.exe scripts\evaluation\run_lora_experiment_a.py --source-job 5f1e2ab448cd4809adad9cd4ac5a8b64 --rank 16 --steps 500
.venv\Scripts\python.exe scripts\evaluation\evaluate_lora_experiment.py --job 27dd93a09bd846c1a08762f95941805e --compare-job 990e32dba22f401eac209582b8743556
```

評估：相同中性 prompt、seed 42/123/314159、LoRA 1.0、CFG 7、30 inference steps。
輸出位於 output/current/lora_rank_comparison。權重僅作實驗，不替換正式面具。
