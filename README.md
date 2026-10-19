# Visual Mask System

以文字情緒推論搭配 **LivePortrait 照片面具**的多頁式 WebSocket 聊天系統。

目前版本不需要為使用者訓練 LoRA，也不以 SD1.5 重繪最終圖片。使用者上傳照片、正方形裁切、確認基底後，即可在聊天室顯示對應表情。

> 這是研究／本機展示系統。自動中性化仍可能出現大小眼、嘴形變形或身分偏移；不能保證任意哭臉、鬼臉或遮擋照片都能還原。請務必驗收預覽。

## 功能

- 註冊、登入、私人聊天室與 MySQL 訊息保存。
- `/chats`：聊天對象列表、目前面具選擇及面具管理入口。
- `/masks`：面具命名、照片上傳、預覽確認、切換及刪除。
- 1:1 正方形裁切，可移動、拉邊角縮放；照片外提供白色留邊。
- 後端檢查裁切座標、圖片大小與單人臉；零人臉時補邊重試，不自動挑選多人照中的其中一人。
- 七種情緒：neutral、joy、sadness、anger、surprise、fear、disgust。
- 情緒判斷有誤的使用者回饋入口；不是每次回饋就立即重新訓練模型。

新增面具只使用本次上傳的照片；不會因為同一帳號或同一人物，就偷偷複製其他面具的衣服與背景。

## 目前生成流程

```text
上傳原照 → EXIF 方向校正 → 正方形裁切／白色補邊 → 單人臉檢查
  ├─ 自然中性照片：保留裁切原照
  └─ 自動中性化：LivePortrait 嘗試去除表情 → 初步身分相似度檢查
                         ↓
                    人工預覽確認 → 獨立面具

聊天文字 → 情緒推論 → WebSocket 更新同一則訊息
  ├─ neutral：直接使用已確認基底
  └─ 其他情緒：本人基底 + 對應表情模板 → LivePortrait → 圖片
```

裁切結果統一為 1024×1024，超出照片的範圍填白，不把長方形硬拉成正方形。
刪除面具採封存，不刪除歷史訊息圖片。舊 LoRA 研究工具仍保留供離線比較，**不是目前網站的訓練或生成依賴**。

## 環境與必要資產

建議使用 Windows、Python 3.10、MySQL／XAMPP、NVIDIA CUDA GPU，並安裝 Git、FFmpeg。
LivePortrait 與 Breeze2 是不同模型環境，建議隔離套件，避免 Transformers／CUDA 相依衝突。

| 程序 | 虛擬環境 | 用途 |
| --- | --- | --- |
| 主網站 | `.venv` | FastAPI、MySQL、照片驗證與推論協調 |
| 情緒服務 | `.venv-erc` | Breeze2，預設連接埠 8010 |
| LivePortrait | `external/LivePortrait/.venv` | GPU 表情轉移，由主網站呼叫 |

**GitHub 不包含模型、使用者照片、資料庫、情緒模板或既有面具。新電腦不能只 clone 就直接生成圖片。**

需另外準備：

1. [官方 LivePortrait](https://github.com/KlingAIResearch/LivePortrait)：放在 `external/LivePortrait`，依其文件安裝相依套件、下載權重，並建立該資料夾內的 `.venv`。Windows 後端預期執行檔為 `external/LivePortrait/.venv/Scripts/python.exe`。
2. 七份可信任的表情模板，路徑詳見 [resources/README.md](resources/README.md)。目前設定檔仍名為 `configs/henry_liveportrait_v1.json`，是歷史命名，不代表新面具共用 Henry 的照片。
3. InsightFace `buffalo_l` 模型，放在 `models/insightface/models/buffalo_l/`；有網路時套件可能自行下載，離線環境應事先備妥。
4. 真實推論需要 Breeze2 checkpoint 及適當的 CUDA PyTorch。請依模型授權及硬體環境準備。

照片驗證使用 InsightFace；這與使用 FaceID 擴散生成是不同用途。模型及第三方資產的授權須另行確認。

## 啟動方式（Windows CMD）

### 1. 安裝主網站環境

```bat
cd /d C:\path\to\Visual-Mask-System
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements-photo-masks.txt
copy .env.example .env
```

InsightFace 在部分 Windows 環境需要 C++ Build Tools 才能安裝。已有可用的 `.venv` 不需要重建。
不要將 `requirements-emotion.txt` 與舊 `requirements.txt` 強行混裝至同一個環境；兩者的 Transformers 版本不同。

### 2. 設定並初始化 MySQL

啟動 MySQL，編輯本機 `.env` 的 `VMS_DB_HOST`、`VMS_DB_PORT`、`VMS_DB_USER`、`VMS_DB_PASSWORD`、`VMS_DB_NAME`。
密碼與資料庫名稱請用自己的設定，不要直接複製他人的值。

```bat
call .venv\Scripts\activate.bat
python scripts\database\init_db.py
```

若遇到 1045，檢查連線帳密；若 XAMPP 啟動失敗，先確認是否有另一個 MySQL 占用 3306，不要直接刪除資料庫。
初始化腳本保留舊資料表與本機 demo 資料的相容邏輯；新使用者可直接由註冊頁建立。

### 3. 啟動情緒推論（另一個 CMD）

先依硬體安裝適當的 PyTorch／CUDA，再安裝情緒服務相依套件。

```bat
cd /d C:\path\to\Visual-Mask-System
python -m venv .venv-erc
call .venv-erc\Scripts\activate.bat
python -m pip install -r requirements-emotion.txt
python emotion_server.py
```

`.env` 的 `VMS_EMOTION_API_URL` 預設是 `http://127.0.0.1:8010`。
若使用內部權杖，兩個程序的 `VMS_EMOTION_API_TOKEN` 必須一致。
模型服務初次載入或下載可能耗時；先確認沒有模型相依錯誤，再啟動網站。

只測介面／關鍵字規則時，可在本機 `.env` 設 `VMS_MOCK=1` 並留空 `VMS_EMOTION_API_URL`，不啟動 Breeze2；**圖片表情處理仍需要 LivePortrait 與 GPU，不是完整無模型模式**。

### 4. 啟動網站

確認 LivePortrait 環境、權重及七種模板已準備好，再開另一個 CMD：

```bat
cd /d C:\path\to\Visual-Mask-System
call .venv\Scripts\activate.bat
python chat_server.py
```

開啟 [本機網站](http://127.0.0.1:8000)，註冊／登入後進入面具管理。
[健康檢查](http://127.0.0.1:8000/health) 應回報 `image_backend: liveportrait`。
舊 `/lora-training` 連結會轉到 `/masks`，不再提供訓練功能。
更新程式後請重啟伺服器，並於瀏覽器按 Ctrl+F5。

## 面具操作

1. 到 `/masks` 輸入名稱並選照片。
2. 移動正方形框、調整大小，可框入照片外白底；白底不能補回已缺失的五官。
3. 選擇「自動中性化」或「自然中性照片，不修改表情」。
4. 新增預覽後，檢查眼睛、嘴角、本人相似度；不合格就刪除草稿並補照片。
5. 勾選驗收並確認，才會啟用；之後在 `/chats` 切換面具。

`neutral` 現在使用固定確認圖；目前尚無同情緒隨機變體。背景也不會自動去背，白色裁切留邊不等於人物分割。

## 測試

以下測試使用暫存圖片、模擬帳號與模擬生成，不需要 MySQL 或載入完整生成模型；仍需安裝主網站的測試相依套件。

```bat
call .venv\Scripts\activate.bat
python -m unittest tests.test_face_validation_retry tests.test_portrait_masks tests.test_portrait_mask_api tests.test_portrait_pipeline
```

前端裁切幾何測試需要 Node.js：

```bat
node tests\test_square_cropper.cjs
node --check web\masks.js
node --check web\chats.js
```

真實 GPU smoke test 為選用，需完整本機資產與一張有權使用的照片；使用隔離面具資料，不改目前面具：

```bat
python scripts\evaluation\smoke_photo_mask.py --source C:\path\to\neutral-photo.jpg
```

## 目錄與資料保存

| 位置 | 用途 |
| --- | --- |
| `src/interfaces/multipage_app.py` | 網頁、API、WebSocket 與權限 |
| `src/services/portrait_masks.py` | 面具建立、裁切、確認、切換與封存 |
| `src/services/pipeline_service.py` | 情緒與 LivePortrait 協調 |
| `src/integrations/liveportrait_expression.py` | 呼叫獨立 LivePortrait 程序 |
| `web/masks.html`、`web/masks.js` | 面具管理介面 |
| `web/square_cropper.js` | 正方形裁切與白底留邊 |
| `data/portrait_masks/` | 私人照片、面具與選擇紀錄，不提交 Git |
| `output/current/` | 生成圖片與歷史訊息引用，不提交 Git |
| `scripts/evaluation/`、`scripts/training/` | 含歷史研究工具，非網站必要操作 |

備份系統時需另外備份 **MySQL、data、output 與本機模型／模板**；Git 分支不是使用者資料備份。
第一次列出面具時，若本機仍有舊任務，系統只匯入其已上傳基底，不讀取 LoRA 權重；舊非正方形基底以等比例補邊相容。

## 隱私、安全與已知限制

- 不提交 `.env`、權杖、個人照片、模型、執行紀錄或資料庫內容。
- 本系統預設只監聽 127.0.0.1；正式部署前需更換簽章密鑰、設定 HTTPS、限制上傳與存取權限。
- 既有 `/output` 靜態圖片路徑未提供逐使用者授權，不應直接暴露至公開網際網路。
- 自動中性化的身分相似度門檻只是初步篩檢，不是身分驗證、準確率或五官不變形保證。
- 人工確認不等於自動品質驗證；極端角度、反光、遮擋、閉眼或誇張表情仍可能失敗。
- 新版面具刪除是可恢復封存，不是立即抹除照片；要做永久刪除需另訂資料保留政策。

