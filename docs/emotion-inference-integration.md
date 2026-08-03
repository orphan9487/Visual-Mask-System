# 獨立情緒推論服務

此服務只執行交接包的文字 ERC 推論。LINE、WebSocket、emoji／貼圖融合、
VisualInstruction 與 LoRA 圖片生成仍由主程式負責。

## 架構

主程式透過 `VMS_EMOTION_API_URL` 呼叫 localhost `/analyze`。未設定 URL 時使用
原本的同程序推論；服務暫時不可用時，`VMS_EMOTION_API_FALLBACK=local` 會回退
本地推論。

## 建立環境

交接包使用 transformers 4.47.1，而主環境使用 5.x，因此不可覆蓋主環境套件。

```powershell
C:\Users\User\anaconda3\envs\mask_env\python.exe -m venv --system-site-packages .venv-erc
.\.venv-erc\Scripts\python.exe -m pip install -r requirements-emotion.txt
```

`--system-site-packages` 只用來共用已安裝的 PyTorch 2.6.0+cu124；
transformers、accelerate、peft 與 bitsandbytes 會在 `.venv-erc` 內使用交接包版本。

## 啟動順序

終端機一（情緒模型，僅綁定 localhost）：

```powershell
.\.venv-erc\Scripts\python.exe -m uvicorn src.interfaces.emotion_inference_app:app --host 127.0.0.1 --port 8010
```

終端機二（現有 LINE/WebSocket）：

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.interfaces.websocket_app:app --host 127.0.0.1 --port 8000
```

主程式 `.env`：

```dotenv
VMS_MOCK=0
VMS_MODEL=breeze2-3b
VMS_QUANT=4bit
VMS_EMOTION_API_URL=http://127.0.0.1:8010
VMS_EMOTION_API_TIMEOUT=120
VMS_EMOTION_API_FALLBACK=local
```

兩個程序會讀取相同模型設定，但 8010 服務明確呼叫本地模型函式，不會再次
呼叫自己。

## 驗證

```powershell
Invoke-RestMethod http://127.0.0.1:8010/health
$body = @{text='我很不開心'; history=@()} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8010/analyze -Method Post -ContentType 'application/json' -Body $body
```

回傳的 `source` 應為 `llm`；若是 `mock` 或 `fallback`，不能視為交接包模型
已成功整合。
