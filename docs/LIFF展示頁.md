# LIFF 展示頁（在 LINE 內開啟的互動 demo）

一頁**只讀**的展示頁，可在 LINE 內（或一般瀏覽器）打開，用來即時展示推理層：
輸入一句話 + 前文 → 看它讀出的情緒、valence、推理鏈與對應的視覺面具。
最適合口試/參賽現場，把「同一句『真好啊』因前文不同而情緒翻轉」包成可點的東西。

- 路由：`GET /liff`（由 `src/line_bot/app.py` 提供，頁面在 `src/line_bot/static/liff.html`）
- 只呼叫既有端點：`/debug/predict`、`/feedback/stats`、`/media/*`，**不寫入任何狀態**
- 沒設定 `LINE_LIFF_ID` 也能用——直接用瀏覽器開 `PUBLIC_BASE_URL/liff`（純網頁模式）

## 頁面內容
1. **互動判讀**：前文（多行，一行一句）+ 這句話 → 情緒徽章 + valence 標籤 + 面具圖 + 推理鏈。
   附兩顆一鍵預設：「😒 反諷版」（前文=刪資料庫）與「😄 真心版」（前文=拿獎學金）。
2. **7 面具庫**：7 情緒 × 素材圖（目前多為佔位圖，待生成層接上換成真面具）。
3. **學習迴圈統計**：即時拉 `/feedback/stats`（互動則數 / 回饋 / 更正樣本）。

## 要在 LINE 內原生開啟才需要的設定（選用）

> **多數情況不需要做這段。** 展示頁本來就是一般網頁：直接用瀏覽器開
> `PUBLIC_BASE_URL/liff`，或把這個網址貼進 LINE 群組點開（走 LINE 內建瀏覽器），
> 判讀/面具庫/統計全都可用。LIFF 只多「用名字打招呼」與「送面具到聊天室」兩個小功能。

⚠️ **LINE 政策改動（2024 起）**：**Messaging API channel 已無法新增 LIFF app**，
必須改用 **LINE Login channel**（Console 會顯示 "Use a LINE Login channel instead"）。
你的 bot 是 Messaging API channel，所以要另開一個 Login channel 來掛 LIFF：

1. Console home → 同一個 provider（例：專題測試）→ **Create a new channel** → 選 **LINE Login**，填基本資料建立。
2. 進該 **LINE Login channel** → **LIFF** 分頁 → **Add**：
   - **Endpoint URL**：`https://<你的-ngrok-或-tunnel-網域>/liff`
   - **Size**：`Full`（整頁）
   - **Scope**：勾 `profile`（用名字打招呼）；若要「送面具到聊天室」再勾 `chat_message.write`
3. 建立後會拿到 `liff-xxxxxxxx-xxxxxxxx`，填進 `.env`：
   ```
   LINE_LIFF_ID=liff-xxxxxxxx-xxxxxxxx
   ```
4. 重啟伺服器。分享 `https://liff.line.me/<liff-id>` 或在 LINE 內開啟即進入。

> LIFF 掛在 Login channel、bot 掛在 Messaging API channel，兩者各自獨立、可並存於同一 provider。
> 通知裡建議的「LINE MINI App」對本 demo 屬殺雞用牛刀，現有/未來的 LIFF app 仍可正常使用。

## 行為說明
- **降級安全**：`LINE_LIFF_ID` 留空、或不在 LINE 環境時，LIFF 初始化靜默略過，頁面仍完整可用。
- **送到聊天室**：僅在 LINE 內（`liff.isInClient`）且有 `chat_message.write` 權限時才顯示按鈕；
  送的是目前情緒的面具圖（用 `PUBLIC_BASE_URL/media/<emotion>.png`，需公開 HTTPS）。

## ⚠️ 安全備註
展示頁呼叫的 `/debug/predict` 目前**無認證**（會對任意輸入跑模型）。
demo 期間可接受，但正式對外前應：關閉 `/debug/*`、或加存取控制、或另開一個受限的 `/api/analyze`。
展示頁本身只讀、不寫狀態，故無 ID token 驗證需求（那是「設定頁」才會碰到的關卡）。
