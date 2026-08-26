# 分層平行協作指南

一份 codebase、一條 `main`;每一「層」在**自己的 git worktree + 分支**上獨立開發,各自開一個
Claude 對話視窗,最後各自開 PR 合回 `main`。

## 分層與負責範圍

| 層 | 目錄(只改這裡) | 分支 | worktree 路徑 |
| --- | --- | --- | --- |
| 感知 perception | `src/perception/` | `layer/perception` | `../vms-layers/perception` |
| 推理 reasoning | `src/reasoning/` | `layer/reasoning` | `../vms-layers/reasoning` |
| 生成 generation | `src/generation/` | `layer/generation` | `../vms-layers/generation` |
| 服務 services | `src/services/` | `layer/services` | `../vms-layers/services` |
| 介面 interfaces | `src/interfaces/` | `layer/interfaces` | `../vms-layers/interfaces` |

`services` 與 `interfaces` 是**整合層**,會呼叫其他層;通常最後合併,或由統籌者維護。

## 每層一個 Claude 對話的做法

1. 在該層的 worktree 目錄開一個終端機,執行 `claude`。
2. 開場白告訴它範圍,例如:
   > 你負責 perception 層。只修改 `src/perception/` 內的檔案。跨層的介面改動要先在 PR 說明,不要直接動別層的檔。先讀 `docs/COLLABORATION.md`。
3. 該對話只在這個 worktree 內工作,不會影響別層。

## 鐵則(避免互相踩線)

- **只改自己層的目錄**。要改共用檔(尤其 `src/services/pipeline_service.py`、各層的 `base.py` /
  介面定義)一律先開 PR 討論,不要在自己分支偷改。
- **層與層之間只透過既有介面溝通**(例如 renderer 的 `base.Renderer`、pipeline 的
  `analyze()/generate()`)。要改介面 = 跨層變更 = 需協調。
- **常常從 main 同步**:`git fetch origin && git merge origin/main`,減少最後的大衝突。
- 完成一段就 **push 分支 + 開 PR**,小步快跑,不要囤一大包。

## 每個 worktree 的環境

worktree 只會有「被追蹤的程式碼」,不含 `models/`(權重)、`data/`、`.venv`(都被 gitignore)。
要在某層實際「跑起來」而不只是改 code,需在該 worktree 內:

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-cuda.txt
copy ..\..\Visual-Mask-System\.env .env      # 沿用你的 .env(含 VMS_PROFILE)
```

純改 code / 寫測試則不需要以上步驟。

## 合併流程

```
layer/<name>  --PR-->  main
```

- 每個 PR 盡量只動自己層的檔,方便 review。
- 合併順序建議:先 perception / reasoning / generation(葉層),再 services / interfaces(整合層)。
- 合併後其他層 `git fetch && merge origin/main` 拿到最新共用碼。

## 建立 / 移除 worktree(統籌者用)

```bash
# 建立(每層一次)
git worktree add ../vms-layers/perception -b layer/perception

# 列出
git worktree list

# 移除(該層工作完、分支已合併後)
git worktree remove ../vms-layers/perception
```
