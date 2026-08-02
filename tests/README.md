# Manual generation tests

These tests exercise GPU-heavy generation workflows and are intended to be run
explicitly from the project root. They are not lightweight unit tests.

```powershell
python tests/manual_instruction_overrides.py
python tests/manual_lora_emotions.py --quick
python tests/manual_verify_mask.py
```
