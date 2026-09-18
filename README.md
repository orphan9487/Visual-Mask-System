# Visual Mask System

自製 WebSocket 聊天室：瀏覽器傳送訊息，系統即時完成情緒推論、建立 VisualInstruction，再由 LoRA 生成情緒面具（圖片／影片）。純本地部署，不含 LINE、talking face、TTS。

## 架構

```text
index.html
   │ WebSocket /ws/{client_name}
   ▼
src/interfaces/websocket_app.py
   │
   ▼
src/services/pipeline_service.py
   ├─ emotion_service.py          情緒與意圖推論
   └─ instruction_runner.py       LoRA 圖片／影片生成
```

`PipelineService` 不依賴 WebSocket。日後要接其他前端時，應新增介面 adapter 呼叫同一個 service，不要把平台邏輯寫進推論或生成層。

## 啟動

1. 建立環境並安裝套件。RTX 4070／CUDA 12.4 使用：

   ```powershell
   pip install -r requirements-cuda.txt
   ```

   不需要 GPU 的 WebSocket／測試環境可使用 `requirements-dev.txt`；訓練與評估工具
   使用 `requirements-training.txt`。單獨安裝 `requirements.txt` 可能取得 CPU 版 Torch，
   因此不適合執行 LoRA 生成。

2. 複製 `.env.example` 為 `.env`，填入 `VMS_DB_*` 資料庫連線設定，然後初始化：

   ```powershell
   python scripts/database/init_db.py
   ```

3. 啟動 WebSocket 伺服器：

   ```powershell
   python chat_server.py
   ```

4. 開啟 `http://127.0.0.1:8000`。健康檢查位於 `http://127.0.0.1:8000/health`。

也可以用 `python main.py "訊息" --mask human` 從命令列直接驗證完整推論與生成流程。

## 測試

不載入模型的核心流程單元測試：

```powershell
python tests/test_pipeline_service.py
```

GPU 生成檢查集中在 `tests/manual_*.py`，不會被一般測試探索自動執行。生成檔統一寫入 `output/current/`。
