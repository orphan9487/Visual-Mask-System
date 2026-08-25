# Visual Mask System

單一 codebase，兩個變體，靠設定檔切換（不再分支分叉）：

| 變體 | `VMS_PROFILE` | 前端 | Talking face / TTS |
| --- | --- | --- | --- |
| 校內專題版 | `thesis`（預設） | 自製 WebSocket 聊天室 | 關 |
| 競賽版 | `competition` | LINE Messaging webhook | 開 |

共同核心（情緒推論、VisualInstruction、LoRA 生成）兩版共用一份。LINE、talking face、TTS 都是可選模組，由 `src/config/features.py` 依 `VMS_PROFILE` 決定是否載入——`thesis` 模式完全不會 import `line_webhook`，因此不需要 `line-bot-sdk` 或 LINE 憑證即可啟動。

## 切換變體

複製對應的設定範本為 `.env`：

```powershell
copy .env.thesis.example .env        # 校內版：自製聊天室
# 或
copy .env.competition.example .env   # 競賽版：LINE + talking face
```

也可在既有 `.env` 裡單獨覆寫：`VMS_ENABLE_LINE`、`VMS_ENABLE_TALKING_FACE`、`VMS_ENABLE_TTS`（值 `0`/`1`）。

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

`PipelineService` 不依賴 WebSocket。日後比賽需要接 LINE 或其他 Hugging Face 系統時，應新增介面 adapter 呼叫同一個 service，不要把平台邏輯寫進推論或生成層。

## 啟動

1. 建立環境並安裝套件。RTX 4070／CUDA 12.4 使用：

   ```powershell
   pip install -r requirements-cuda.txt
   ```

   不需要 GPU 的 WebSocket／測試環境可使用 `requirements-dev.txt`；訓練與評估工具
   使用 `requirements-training.txt`。單獨安裝 `requirements.txt` 可能取得 CPU 版 Torch，
   因此不適合執行 LoRA 生成。

2. 依變體複製設定範本為 `.env`（見上方「切換變體」），填入 `VMS_DB_*` 資料庫連線設定（競賽版另需 `LINE_*` 憑證），然後初始化：

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
