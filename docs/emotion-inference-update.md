# 情緒推論更新

整合來源：orphan9487/Visual-Mask-System，896737fa27981161d4c1677cb380cd01bc672cfc。

預設保留 Breeze2 底座與 two_stage 推論。下列為可選 .env 設定：

```env
VMS_HISTORY_TURNS=20
VMS_ERC_MODE=two_stage
# VMS_LORA_PATH=models/breeze2_erc_cped80series_sarcasm_lora
# VMS_ERC_CALIB_ALPHA=0.5
```

只有持有相容的情緒模型 adapter 時才設定 VMS_LORA_PATH；這不是人臉 LoRA。
校正只在 Breeze2、single 模式且 adapter 名稱符合 configs/erc_calibration.json 時套用。
校正強度設為 0 可停用。預設沒有 adapter 時不套用該校正。

私人聊天室只保留真實發言及發話者，每房間獨立，最近 20 則。
兩個服務均須重啟：.venv-erc 執行情緒服務，.venv 執行聊天服務。

獨立推論服務新增 POST /explain；新版聊天室提供經登入和房間權限檢查的
POST /api/chats/{room_id}/messages/{message_id}/explain。
目前僅能解釋本次伺服器啟動後有保留語境的訊息，介面尚未加入按鈕。
說明是事後產生，不是判讀當下的推理紀錄，也不更改原標籤。

Hybrid、影像 LoRA 訓練與回饋資料儲存保持原流程。
