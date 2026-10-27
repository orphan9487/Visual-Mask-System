# 繁中對話情緒標註工具（產品 benchmark）

把 `docs/標註schema_草案v0.md` 轉成 code。這是「資料護城河」的地基：
用**對的標註**在「人類能穩定同意 ∧ GPT/Claude 會錯」的交集上，做出大廠買不到的評測集。

> **承重牆禁忌**：不能用一個你聲稱會錯的模型，去產生你的 ground truth。
> 模型只能**初標**，決定對錯的最後一關必須是**人逐條改**（每條過人腦，禁止抽查）。

## 模組
| 檔案 | 做什麼 |
|---|---|
| `schema.py` | 資料結構（Conversation/Utterance/Annotation/Cause）+ 驗證 + 距離自動算 + JSONL 讀寫 |
| `pre_annotate.py` | 初標助手：用 ERC 引擎/mock 對每則產 `_draft` 草稿，供人改 |
| `iaa.py` | 標註者間一致性：Fleiss' κ（valence/情緒）+ 逐標籤 κ/Jaccard + 成因命中率 + 多數決 gold 建議 |

標籤空間（**與 `src/reasoning/labels.py` 競賽 7 類刻意分開**）：
- **Tier1 valence**（必標、承重牆）：positive / negative / neutral / ambiguous
- **Tier2 情緒**（單標）：neutral, joy, anger, contempt, sadness, anxiety, surprise, resignation, ambiguous
- **難度標籤**（多選）：sarcasm, code_switch, net_slang, distant_cause
- **成因**：source(in_conversation/external/unclear) + cause_uid + distance(自動) + note

## 流程

**1. 準備原始對話**（去識別化 + 取得同意）→ JSONL，每行一段：
```json
{"conv_id":"c001","meta":{"source":"line","consent":true,"deidentified":true,"speakers":["S1","S2"]},
 "utterances":[{"uid":0,"speaker":"S1","text":"..."}, ...]}
```

**2. 初標**（先 mock 驗管線，再上真模型）：
```bash
# 先驗管線（不載入模型）
python -m src.annotation.pre_annotate --in data/annotation/pilot_raw.jsonl \
       --out data/annotation/pilot_draft.jsonl --mock
# 正式（Breeze2-3B，兩階段有 rationale 才會有 sarcasm 草稿）
python -m src.annotation.pre_annotate --in data/annotation/pilot_raw.jsonl \
       --out data/annotation/pilot_draft.jsonl --model breeze2-3b --quant fp16
```
輸出每則含 `_draft`（模型建議）+ `_draft_meta`（出處/推理鏈），`per_annotator:[]`、`gold:null`。
草稿只自動打客觀的詞彙級難度標籤（`code_switch`/`net_slang`）；**`sarcasm` 與 `cause` 刻意留空**
——反諷正是模型的盲點（用會錯的模型去初標反諷邏輯自相矛盾），成因則是偏誤風險最高的欄位，
兩者都由人從零判。

**3. 人逐條改**：≥3 人各自把 `_draft` 改成自己的判斷，填進 `per_annotator`；分歧討論裁決成 `gold`。

**4. 隨時驗證格式**：
```bash
python -m src.annotation.schema data/annotation/pilot_draft.jsonl          # 檢查 schema
python -m src.annotation.schema data/annotation/pilot_gold.jsonl --require-gold
```

**5. 算 IAA**（pilot 標完、有多人 per_annotator 後）：
```bash
python -m src.annotation.iaa data/annotation/pilot_labeled.jsonl
# 順便用多數決產 gold 建議（寫 _gold_suggestion，不覆蓋 gold）
python -m src.annotation.iaa data/annotation/pilot_labeled.jsonl --suggest-gold data/annotation/pilot_gold_suggested.jsonl
python -m src.annotation.iaa --selftest   # 用 Fleiss 標準例驗證數學
```
→ pilot 決策點：valence κ 達 substantial(≥.6)＝承重牆穩；情緒/反諷 κ 達 moderate 以上可用；
成因 κ 或 cause_uid 命中率太低 → v1 砍成因。**一致性只證「人能穩定同意」；
還要配 Claude 尺證「模型會錯」才成立護城河**（Claude 尺＝下一個工具，需你自備 API 金鑰）。

**6.（下一步）Claude 尺**：把同一批 gold 餵給 Claude/GPT API，按難度標籤拆解錯誤率，
量化「模型在反諷/遠距成因上錯多少」——與 IAA 兩相對照即護城河證據。

## 自動檢核（schema.py 幫你抓的錯）
- valence ↔ emotion 一致性（joy 必 positive、anger/contempt/sadness/anxiety/resignation 必 negative；surprise/ambiguous 自由）
- 中性訊息不可有 cause；in_conversation 成因的 cause_uid 必須存在且不在未來
- 距離 = uid − cause_uid 自動算；距離 ≥2 自動要求 distant_cause 標籤（打了/沒打不一致會報錯）
- 標籤合法性、重複、uid 唯一、n_utterances 相符

## 檔案慣例
- `_example_pipeline_demo.jsonl`：**僅供管線示範**，含反諷殺手案例。**不是評測 gold**，
  真實對話（=評測尺）另存、需 consent。合成/示範資料絕不當 ground truth。

## 真實聊天測試集（7 類，評估部署模型用）

與上面的產品細層 schema 分開：這裡用部署模型同一套 7 類標籤，目的是量「模型在真實台灣聊天上的表現」與「人類彼此一致度（天花板）」。

1. **收語料**（只收參與者知情同意的對話，檔案放 `data/chat_testset/`，不進 git）
   - 聊天室錄製：`.env` 設 `VMS_CHAT_RECORD=data/chat_testset/recordings.jsonl` 後啟動聊天室，每則發言會被記下
   - 或純文字逐字稿：每行 `名字：內容`，空行或 `---` 分隔不同對話
2. **整理**：`python -m src.annotation.chat_testset build --record data/chat_testset/recordings.jsonl --txt "data/chat_testset/raw/*.txt"`
   發話者自動匿名成 A/B/C，電話/email/網址/身分證字號自動遮蔽；發出前仍要人工瀏覽一遍
3. **標註**：兩人各自用瀏覽器開 `src/annotation/chat_annotator.html`，載入 `items.jsonl`，標完匯出 `labels_<名字>.json`（互不討論）
4. **一致度＋分歧清單**：`python -m src.annotation.chat_testset agree data/chat_testset/labels_*.json --adjudicate data/chat_testset/adjudicate.jsonl`
   兩人討論分歧題，在 `final` 欄填最後的標籤
5. **合併 gold**：`python -m src.annotation.chat_testset gold data/chat_testset/labels_*.json --adjudicate data/chat_testset/adjudicate.jsonl`
6. **評估模型**：`python -m src.evaluation.eval_chat_testset --annotators data/chat_testset/labels_*.json`
   與上線設定相同（單次打分＋先驗校正＋20 句語境），並列出模型 vs 人、人 vs 人的 κ

建議規模：300–500 則（約 30–50 段對話）。少於 200 則時 CI 會很寬，只能看大方向。

### 論壇補充子集（PTT 推文，主要測反諷／酸文）

- 抓取：`python -m src.annotation.ptt_crawler --out data/chat_testset/forum_raw.jsonl`（預設 Boy-Girl、WomenTalk、marriage、Tech_Job、joke，每板 6 篇、每篇最多 15 則推文）
  - 只抓公開看板；需要年齡確認的看板（如 Gossiping）自動跳過；請求間隔至少 1 秒
  - 純附和的推文（「推」「+1」）不收；推／噓標記不交給標註者與模型
- 整理：`python -m src.annotation.chat_testset build --forum data/chat_testset/forum_raw.jsonl --out data/chat_testset/forum_items.jsonl`
  - 文章標題＋內文只當語境（標註頁顯示「原文」），標註對象是推文；帳號與內文提到的帳號都換成代號
- 論壇和聊天是不同領域，**分開評估、分開報告**，不要合成同一個分數
