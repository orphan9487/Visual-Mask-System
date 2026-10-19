# Neutralization region comparison

Two user-provided portraits, each tested with absolute full-expression (`exp`)
and absolute lips-only (`lip`) transfer. Same neutral template, multiplier 1.0,
relative motion off, lip normalization off. No SD1.5, LoRA, database mutation,
or active base replacement. Standalone harness overrides the subprocess argument
only for this run; production settings are unchanged.

Command: `.venv\Scripts\python.exe scripts/evaluation/compare_neutral_regions.py`

Outputs: `output/current/neutral_region_comparison/` (four generated images,
full-image contact sheet, and source/region/output report).

Visual observations, not quantitative eye-symmetry measurements:

- Full-expression transfer changes eyelid opening on both subjects; subject 2
  shows a visibly different opening between eyes.
- Lips-only better preserves the original narrowed eyes, but leaves smile-eye
  cues on subject 1 and half-closed eyes on subject 2. Neither is a complete
  neutralization solution.
- Both modes close the mouth, with residual lip tension/shape artifacts.
- No evidence yet proves that the template alone is the cause: pose, glasses,
  landmark estimation, and neural reconstruction remain possible contributors.
- Next test should separate lip neutralization from restrained per-subject eye
  retargeting. Avoid forcing perfectly symmetric eyes; compare against source
  geometry and inspect glasses/identity preservation. A naturally neutral source
  portrait remains the lowest-edit alternative.

## Follow-up: restrained eye retargeting

Ran `scripts/evaluation/compare_neutral_eye_control.py` with the LivePortrait
virtual environment. Two portraits, lips-only absolute transfer, eye target
increments 0 / 0.02 / 0.04, correction blend 0.5: six completed outputs.
Subtracting the network's baseline-eye delta isolates the incremental correction;
correction is applied before stitching. No production code/settings are changed.

Important limitation: the pretrained eye network takes two source-eye ratios but
only one target-eye scalar. This is source-conditioned control, NOT independent
left/right calibration or a guarantee of preserved natural asymmetry.

Source landmark eye ratios were approximately [0.1626, 0.1911] and
[0.1274, 0.1059]. These are model-estimated opening ratios, not medical or
anatomical measurements. Glasses and pose can bias them.

Visual comparison shows only small opening changes; half-closed/smile-eye cues
and residual lip tension remain. No reliable correction of eye asymmetry has
been demonstrated. Do not promote these settings to production based on this
test. Outputs/report: `output/current/neutral_eye_control/`.
