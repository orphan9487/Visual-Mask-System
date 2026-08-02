# Project scripts

Standalone maintenance and experiment commands are grouped by purpose:

- `training/`: dataset preparation and model training.
- `generation/`: controlled image-generation runs.
- `evaluation/`: model and identity comparisons.
- `reporting/`: reproducible figures and PDF reports.
- `database/`: legacy database initialization.

Run commands from the project root, for example:

```powershell
python scripts/generation/generate_emotion_candidates.py
python scripts/reporting/report_pipeline_visual.py
```

Scripts resolve project resources from their own location, so their behavior
does not depend on the current working directory.
