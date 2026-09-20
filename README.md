# Visual Mask System

這個分支是系統的純 WebSocket + FaceID 版本。瀏覽器透過 FastAPI WebSocket 傳送訊息，系統即時完成情緒推論、建立 VisualInstruction，再由 FaceID 或保留的 LoRA 後端生成圖片。執行期不包含 LINE Bot、LINE webhook、LINE SDK、Talking Face 或 TTS。

## 架構

```text
index.html
   │ WebSocket /ws/{client_name}
   ▼
src/interfaces/websocket_app.py
   │
   ▼
src/services/pipeline_service.py
   ├─ emotion_service.py          情緒、複合情緒與意圖推論
   ├─ emotion_inference_client.py 可選的獨立 Breeze2 服務
   └─ instruction_runner.py       FaceID 靜態圖片／LoRA 備援生成
```

`PipelineService` 不依賴 WebSocket。FaceID 身分會固定輸出 PNG，不會進入 AnimateDiff。LoRA renderer 仍保留，方便比較或回復舊方案。

聊天室會保留一個主要情緒供 FaceID 生成，並在文字與 emoji 衝突時額外顯示
次要情緒及複合名稱（例如 sadness + joy →「口是心非」）。這些呈現資訊不會
改變既有的主要情緒標籤，也不會破壞生成層介面。

## 可選的獨立情緒推論服務

預設仍在 WebSocket 程序內執行情緒推論。若 Breeze2 與圖片生成環境的
`transformers` 版本衝突，可建立第二個虛擬環境：

```powershell
python -m venv .venv-erc
.venv-erc\Scripts\python.exe -m pip install -r requirements-emotion.txt
$env:VMS_EMOTION_API_TOKEN="請設定一組內部權杖"
.venv-erc\Scripts\python.exe -m uvicorn src.interfaces.emotion_inference_app:app --host 127.0.0.1 --port 8010
```

再於主系統 `.env` 設定：

```dotenv
VMS_EMOTION_API_URL=http://127.0.0.1:8010
VMS_EMOTION_API_TOKEN=請設定相同權杖
VMS_EMOTION_API_TIMEOUT=120
VMS_EMOTION_API_FALLBACK=local
```

URL 留空時完全沿用原本的本機推論。健康檢查會以 `emotion_inference` 欄位
回報目前使用 `local` 或 `remote`。

## FaceID 設定

1. 安裝 FaceID 環境：

   ```powershell
   pip install -r requirements-faceid.txt
   ```

2. 將 1～5 張同一人的清晰參考照放進 `data/faceid_profiles/henry/`。這個資料夾已被 Git 忽略，不會把個人照片提交到版本庫。也可以在 `.env` 用分號設定絕對路徑：

   ```dotenv
   VMS_HENRY_FACEID_REFERENCES=C:\path\face1.jpg;C:\path\face2.jpg
   ```

3. WebSocket 系統預設使用 Henry 的 `IP-Adapter FaceID Portrait v11`，強度為
   `0.55`。三個 demo 帳號初始化後都會選取 `Henry FaceID`；其他既有 LoRA
   仍保留在面具選單作為備援與比較用途。

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

   這個初始化指令可重複執行，也會把既有 demo 帳號從舊的 `human_8692`
   active 設定遷移到 `henry` FaceID。

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
