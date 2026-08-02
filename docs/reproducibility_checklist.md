# 純 WebSocket 基線驗收紀錄

驗收日期：2026-08-02

## 已通過

- 從零建立 `.venv`，成功安裝 `requirements-dev.txt`。
- `pip check`：無相依衝突。
- 7 個自動測試通過，包含 Pipeline、資料庫環境設定、健康端點與雙 WebSocket 客戶端廣播。
- FastAPI 應用與 13 條路由可在乾淨環境載入。
- RTX 4070 SUPER 上成功載入本機 SD1.5、fp32 VAE 與 `person8692` LoRA。
- 使用固定 seed `42`、12 inference steps 產生可辨識的 joy 人像。
- 執行期程式沒有 `layers/`、`reasoning_engine.py`、LINE Bot 或 webhook import。

## 指令

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
$env:VMS_MOCK = "1"
.venv\Scripts\python -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python -m pip check
```

GPU 生成環境使用：

```powershell
pip install -r requirements-cuda.txt
```

## 外部設定

本機 MySQL 已回應連線，但拒絕空密碼的 `root`。在 `.env` 填入正確的
`VMS_DB_USER` 與 `VMS_DB_PASSWORD` 後，執行：

```powershell
python scripts/database/init_db.py
python chat_server.py
```

資料庫密碼不應寫入程式碼或提交至 Git。
