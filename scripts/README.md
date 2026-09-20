# Project scripts

Standalone maintenance and experiment commands are grouped by purpose:

- `training/`: dataset preparation and model training.
- `generation/`: controlled image-generation runs.
- `evaluation/`: model and identity comparisons.
- `reporting/`: reproducible figures and PDF reports; `generate_legacy_tech_doc.py`
  only rebuilds the archived pre-WebSocket report.
- `database/`: environment-configured MySQL initialization.
- `learning/`: export MySQL emotion corrections for analysis or future ERC training.

Run commands from the project root, for example:

```powershell
python scripts/generation/generate_emotion_candidates.py
python scripts/reporting/report_pipeline_visual.py
python -m scripts.learning.export_feedback_sft --out data/erc_sft/feedback.jsonl
```

Scripts resolve project resources from their own location, so their behavior
does not depend on the current working directory.
