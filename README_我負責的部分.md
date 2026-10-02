# Visual Mask System — 我負責的部分

> 輸入感知層 · 推理層 · 學習迴圈（含微調與評估管線）

本文件說明我在 Visual Mask System 中負責的三層。整個系統的介紹、架構圖與作品定位見根目錄 [`README.md`](README.md)；生成層由隊友負責（見 [`docs/隊友_生成層待辦.md`](docs/隊友_生成層待辦.md)）。

Visual Mask System 為純文字社交對話**即時判讀情緒與意圖**，並生成個人化的視覺回饋，補償文字流失的情緒訊號。**我負責「從訊息到理解」的完整鏈路**——把一句話（含 emoji/貼圖與對話語境）判讀成情緒與意圖，再產出一份給生成層消費的視覺指令；並建立會從使用者回饋學習的迴圈。

---

## 我在整體系統中的位置

```
使用者訊息 ──▶ ①輸入感知層 ──▶ ②推理層 ──[VisualInstruction JSON]──▶ 生成層(隊友) ──▶ 回覆
                                    ▲                                              │
                                    └──────────── ④學習迴圈 ◀──────────────────────┘
```

我負責 ①②④；與生成層以一份 `VisualInstruction` JSON **合約解耦**，可各自獨立開發（合約見 [`docs/推理層-生成層_介面合約.md`](docs/推理層-生成層_介面合約.md)）。

---

## 模組地圖

### ① 輸入感知層　`src/perception/`
| 檔案 | 職責 |
|---|---|
| `multimodal_fusion.py` | 融合「文字情緒」與「emoji/貼圖情緒」；文字漏掉時由符號補回（如「沒事啊😭」→ 悲傷），文字與符號衝突時標記反諷 |
| `emoji_emotion.py` | 把 emoji / 貼圖 keywords 編碼成情緒訊號（查表，不需重型視覺模型） |
| `summarizer.py` | 文字摘要（BART），供長對話壓縮（元件，選用） |

### ② 推理層　`src/reasoning/`
| 檔案 | 職責 |
|---|---|
| `erc_engine.py` | **Affective Analyzer**：兩階段 M-CoT 對話情緒辨識（ERC），輸出 7 類情緒 |
| `intent_decoder.py` | **Semantic Intent Decoder**：溝通意圖 + 情緒成因 + 對象 + 真誠/反諷 |
| `visual_instruction.py` | **Visual Instruction Generator + Modality Selector**：把情緒轉成給生成層的結構化指令（prompt、影片/貼圖、生成參數） |
| `identity_db.py` | **Identity-DB**：使用者 → 個人化面具（LoRA）綁定 |
| `backbone.py` | 可設定的 LLM 後端（Breeze2 / Qwen2.5 / …，支援 4-bit / fp16） |
| `prompts.py` | 系統/語境/標籤三段 prompt 的單一真實來源（訓練與推論共用，杜絕不一致） |
| `labels.py` | 7 類 canonical 情緒 + 各資料集映射 + 中英情緒詞解析 |
| `try_erc.py` | 互動式測試器（見下） |

### ④ 學習迴圈　`src/learning/`
| 檔案 | 職責 |
|---|---|
| `feedback_monitor.py` | 記錄互動、偵測使用者回饋（讚同/否定/更正），並把「更正」轉為**可再訓練的 SFT 樣本** → 閉合「回饋 → 再訓練」迴圈 |

### 微調與評估　`src/training/`、`src/evaluation/`
LoRA 微調管線（資料格式化、completion-only masking 訓練）、ERC 指標評估（Weighted-F1、幻覺率、消融）、生成一致性評估、資料集檢視。評估輸出見 `evaluation_results/`。

---

## 快速體驗

安裝環境見根目錄 README 或 `requirements.txt`。

```bash
# 互動式測試推理層：看兩階段推理鏈、語境如何改變判讀
.venv\Scripts\python -m src.reasoning.try_erc --model breeze2-3b --quant fp16
#   輸入「你把我的資料庫刪光了」再輸入「真好啊」，觀察情緒隨語境改變

# 重現「情緒微調前後」實驗（繁中對話 CPED）
.venv\Scripts\python -m src.training.format_meld --dataset cped --split train --limit 30000
.venv\Scripts\python -m src.training.train_erc_lora --data data/erc_sft/cped_train.jsonl --out models/qwen_erc_cped_lora
.venv\Scripts\python -m src.evaluation.eval_erc --model qwen2.5-1.5b --lora models/qwen_erc_cped_lora --single --dataset cped --limit 300
```

---

## 關鍵成果（我的驗證）

- **微調有效（跨語言/形式）**：MELD 0.331→0.538、繁中單句 0.585→0.868、繁中對話 CPED 0.189→0.345（Weighted-F1）
- **語境的乾淨貢獻**：2×2 消融證明 **+0.151**（0.194→0.345）
- **資料規模效應**：CPED 0.189→0.261→0.345（隨在地資料單調提升、未飽和）
- **誠實發現**：兩階段 M-CoT 在 1.5B 小模型上不提升準確率、但降低幻覺率——以數據精確化文獻主張
- **端到端整合驗證**：12/12 通過（感知→推理→融合→指令→學習迴圈）

（實驗設計與完整數據見 [`docs/計畫書_五_實驗設計與結果.md`](docs/計畫書_五_實驗設計與結果.md)。）

---

## 與生成層的介面

我的推理層輸出一份 `VisualInstruction` JSON，生成層照其 `positive_prompt` / `negative_prompt` / `identity.lora_path` / `modality` / `generation_hints` 生成影像。欄位與消費範例見 [`docs/推理層-生成層_介面合約.md`](docs/推理層-生成層_介面合約.md)。

## 技術棧

Python 3.12、PyTorch (CUDA)、Transformers、PEFT (LoRA)、FastAPI、SQLite、emoji、OpenCC。底座 LLM：Llama-Breeze2-3B（繁中）／ Qwen2.5。單張消費級 GPU（RTX 4070）即可訓練與部署。
