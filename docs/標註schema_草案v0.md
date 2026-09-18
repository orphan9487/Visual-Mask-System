# 繁中對話情緒標註 Schema 草案 v0

供兩位工程 + 標註者對齊。這份是**待審草案**，鎖定前先跑 20 條 pilot 用 IAA 驗證。

## 0. 設計原則（回顧，所有取捨從這推）

- **北極星**：每個類別/欄位，最大化「人類能穩定同意 ∧ GPT 會錯」的交集。
- **兩層情緒**：粗層 valence（高 IAA，承重牆主張建在這）+ 細層情緒（IAA 撐得住才用）。
- **反諷不是情緒，是難度標籤**；反諷句標的是它的**底層真實情緒**。
- **成因只到句級、單一最直接因**；記錄距離；允許「對話外/不明」。
- **允許逃生口**（曖昧/需更多語境），逼標會製造噪音。

---

## 1. 標註單位與資料結構

- **標註單位**：一段對話（conversation）內的**每一則訊息（utterance）**。
- **多標註者**：每則由 ≥3 人獨立標 → 多數決 → 分歧時討論裁決（gold）。
- **格式**：JSONL，一行一段對話。

```
conversation
├─ meta            對話層 metadata（去識別化、來源、同意）
├─ utterances[]    每則訊息（speaker 匿名、text 去識別化）
└─ annotations     每則的標註：各標註者 + gold（裁決後）
```

---

## 2. Tier 1 — Valence（粗層，承重牆）

**必標。** 高 IAA 骨幹，你最穩的護城河主張建在這層。

| 值 | 中文 | 說明 |
|---|---|---|
| `positive` | 正向 | 真實情緒為正（不看字面，看底層） |
| `negative` | 負向 | 真實情緒為負 |
| `neutral` | 中性 | 無明顯情緒/純資訊 |
| `ambiguous` | 曖昧 | 缺語境、真的判不出（逃生口，獨立統計） |

> 反諷的「真好啊」→ valence = `negative`（字面正、底層負）。這一格人類會一致、GPT 常錯 → 這就是你的核心 demo。

---

## 3. Tier 2 — 細層情緒（IAA 撐得住才正式用）

**單標必填，不設次標。** 供更細的差異化；若 pilot 顯示某類 IAA 太低則合併/降級。

| 值 | 中文 | valence | 備註 |
|---|---|---|---|
| `neutral` | 中性 | neutral | |
| `joy` | 開心 | positive | |
| `anger` | 生氣 | negative | 針對**事**、想要你改、**熱** |
| `contempt` | 不屑嘲諷 | negative | 針對**人**、懶得理你、**冷**（居高臨下、呵呵/笑死） |
| `sadness` | 難過 | negative | |
| `anxiety` | 焦慮擔心 | negative | 擔憂、緊張、不安 |
| `surprise` | 驚訝 | 依情境 | 正負皆可 |
| `resignation` | 無奈 | negative | 認命、算了、擺爛（≠難過：難過是失落，無奈是放棄） |
| `ambiguous` | 曖昧 | — | 逃生口 |

（`disgust 噁心` 已砍——真實聊天稀有、且與 `contempt` 易混。）

**共現規則（單標）**：兩情緒都在時，用「事/人、熱/冷」規則**擇一**——主要針對事、想要你改 → `anger`；針對人、居高臨下 → `contempt`。不設次標，共現一律靠此規則單選。

---

## 4. 難度標籤（多選，護城河的量化指標）

**每則可標 0～多個。** 這些是「GPT 尺」拆解錯誤率的依據——能算出「GPT 在反諷案例錯 X%」。

| 值 | 中文 | 定義 |
|---|---|---|
| `sarcasm` | 反諷/言外之意 | **不能照字面理解**：反諷（字面相反，真好啊/你好棒棒）**或**言外之意（隱含、需推論的情緒）。合併原 implicit。 |
| `code_switch` | 語碼混用 | 中英夾雜、台語/客語夾雜 |
| `net_slang` | 注音火星文/網路梗 | ㄏㄏ、484、母湯、需群組共識的梗 |
| `distant_cause` | 遠距成因 | 情緒的成因在 ≥2 則之前（見第 5 節距離） |

---

## 5. 成因（TECPE 簡化版）

**只對「非中性」訊息標。** 中性訊息 cause = null。

每則情緒訊息標一個 cause 物件：

| 欄位 | 值 | 說明 |
|---|---|---|
| `source` | `in_conversation` / `external` / `unclear` | 成因在對話內 / 對話外(考試放榜之類) / 判不出 |
| `cause_uid` | 整數 or null | source=in_conversation 時，成因是「第幾則」（單一最直接因） |
| `distance` | 整數（自動算） | 當前 uid − cause_uid；≥2 → 自動打 `distant_cause` 標籤 |
| `note` | 字串（選填） | 成因是對話外時的簡述（去識別化） |

> **v1 可先不做成因**：若 pilot 發現成因 IAA 太低（SemEval 冠軍才 0.32–0.38），v1 先只做「情緒兩層 + 難度標籤」，成因延到 v2。承重牆很可能光靠情緒+反諷就秀得出來。

---

## 6. 完整 JSONL 範例（含殺手案例）

一段對話一行。以下展開易讀（實際存成單行）：

```json
{
  "conv_id": "c0007",
  "meta": {
    "source": "line",
    "consent": true,
    "deidentified": true,
    "n_utterances": 4,
    "speakers": ["S1", "S2"]
  },
  "utterances": [
    {"uid": 0, "speaker": "S1", "text": "我熬夜三天趕的報告終於好了"},
    {"uid": 1, "speaker": "S2", "text": "喔對 我剛剛不小心把你那個資料夾刪掉了"},
    {"uid": 2, "speaker": "S1", "text": "..."},
    {"uid": 3, "speaker": "S1", "text": "真好啊 謝謝你喔"}
  ],
  "annotations": {
    "3": {
      "per_annotator": [
        {"annotator": "A", "valence": "negative", "emotion_primary": "contempt",
         "difficulty_tags": ["sarcasm", "distant_cause"],
         "cause": {"source": "in_conversation", "cause_uid": 1}},
        {"annotator": "B", "valence": "negative", "emotion_primary": "anger",
         "difficulty_tags": ["sarcasm", "distant_cause"],
         "cause": {"source": "in_conversation", "cause_uid": 1}},
        {"annotator": "C", "valence": "negative", "emotion_primary": "contempt",
         "difficulty_tags": ["sarcasm", "distant_cause"],
         "cause": {"source": "in_conversation", "cause_uid": 1}}
      ],
      "gold": {
        "valence": "negative",
        "emotion_primary": "contempt",
        "difficulty_tags": ["sarcasm", "distant_cause"],
        "cause": {"source": "in_conversation", "cause_uid": 1, "distance": 2},
        "agreement": {"valence": 1.0, "emotion_primary": 0.67}
      }
    }
  }
}
```

**這個範例展示了什麼**：
- Valence 三人全同意 `negative`（IAA=1.0）← 承重牆穩在這
- 細層 A/C=contempt、B=anger、多數決=contempt（IAA=0.67）← 「不屑 vs 生氣」的分歧，正是要 pilot 觀察的
- 難度標籤：`sarcasm`（反諷/言外之意）+ `distant_cause`（成因在第 1 則、距離 2）
- 成因指向 uid=1「刪掉資料夾」——GPT 若只看第 3 則「真好啊」會判 positive，這就是護城河

---

## 7. 標註流程與 IAA

1. 模型/API **初標**（草稿）→ 人**逐條改**（每條過人腦，禁止抽查）
2. ≥3 人獨立標 → 多數決 → 分歧討論裁決成 gold
3. **算 IAA**：
   - Valence / 情緒主標：Fleiss' kappa（多標註者）
   - 難度標籤（多選）：逐標籤 kappa 或 Jaccard
   - 成因：cause_uid 完全命中率（+ 允許 ±1 的寬鬆版）
4. 記錄「GPT/Claude 在哪類 case 系統性錯」清單 ← 行銷資產

---

## 8. v1 先做 / v2 再做

| 欄位 | v1（先做） | v2（後補） |
|---|---|---|
| Valence（粗層） | ✅ 必做，承重牆 | |
| 細層情緒主標（單標，7類+曖昧） | ✅ | |
| 難度標籤 | ✅ 護城河指標 | |
| 成因 | ⚠️ 視 pilot IAA 決定 | 若 v1 IAA 低則延到這 |

**pilot 決策點**：標 20 條 → 若 valence + 情緒 + 反諷標籤的 IAA 可接受、且 GPT/Claude 在這些上明顯錯 → 承重牆成立，放大到幾百條並保留成因；若成因 IAA 太低 → v1 砍成因。
