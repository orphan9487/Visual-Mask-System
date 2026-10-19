# 獨立 Breeze2 情緒推論服務

情緒推論與 WebSocket／FaceID 圖片生成使用不同 Python 環境與程序，避免
Breeze2 所需的 Transformers 版本覆蓋主環境套件。

## 架構

```text
瀏覽器 WebSocket
       |
       v
chat_server.py :8000  ---- HTTP /analyze ---->  emotion_server.py :8010
       |                                         Breeze2 3B (4-bit)
       v
SD1.5 + FaceID
```

兩個服務只綁定 `127.0.0.1`。情緒服務不包含 LINE、WebSocket 或圖片生成。

## 首次建立情緒環境

在專案根目錄執行：

```cmd
C:\Users\User\anaconda3\envs\mask_env\python.exe -m venv --system-site-packages .venv-erc
.venv-erc\Scripts\python.exe -m pip install -r requirements-emotion.txt
```

`--system-site-packages` 用來共用既有的 PyTorch 2.6.0+cu124；Transformers、
Accelerate、PEFT 與 bitsandbytes 會使用 `.venv-erc` 內的相容版本。

## `.env` 設定

```dotenv
VMS_MOCK=0
VMS_MODEL=breeze2-3b
VMS_QUANT=4bit
VMS_EMOTION_API_URL=http://127.0.0.1:8010
VMS_EMOTION_API_TIMEOUT=120
VMS_EMOTION_API_FALLBACK=error
```

`error` 表示 8010 無法使用時直接回報錯誤，避免主程序載入第二份 Breeze2
或悄悄退回關鍵字規則。

## 每次啟動

開啟兩個 CMD，順序如下。

終端機一（等待顯示「模型就緒」）：

```cmd
.venv-erc\Scripts\python.exe emotion_server.py
```

終端機二：

```cmd
.venv\Scripts\python.exe chat_server.py
```

瀏覽器開啟 `http://127.0.0.1:8000`。

## 驗證

```cmd
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8000/health
```

8010 回應的 `configured_model` 應為 `breeze2-3b`、`model_loaded` 應為
`true`；8000 回應的 `emotion_inference` 應為 `remote`。實際分析結果的
`source` 應為 `llm`，不能是 `mock` 或 `fallback`。
