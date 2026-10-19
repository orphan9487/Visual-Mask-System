"""LivePortrait 暫存（含原始照片複本）不得放在公開的 output/，且用完即刪。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.integrations import liveportrait_expression as lp


class LivePortraitRuntimeTests(unittest.TestCase):
    def test_runtime_is_private_and_removed_after_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "face.png"
            Image.new("RGB", (64, 64)).save(source)
            fake_root = tmp / "LivePortrait"
            (fake_root / ".venv" / "Scripts").mkdir(parents=True)
            (fake_root / ".venv" / "Scripts" / "python.exe").write_text("")
            (fake_root / "inference.py").write_text("")
            seen = {}

            def fake_run(command, cwd, check, timeout, env):
                out = Path(command[command.index("--output-dir") + 1])
                staged = Path(command[command.index("--source") + 1])
                seen["runtime"] = out
                seen["staged_exists"] = staged.is_file()
                Image.new("RGB", (64, 64)).save(out / f"{staged.stem}--driving.jpg")

            profile = {"profiles": {"joy": {"template": str(tmp / "driving.mp4"),
                                            "cache": str(tmp / "cache.png")}}}
            with (
                patch.object(lp, "OUT", tmp / "output" / "current" / "instruction_runs"),
                patch.object(lp, "RUNTIME_ROOT", tmp / "data" / "liveportrait_runtime"),
                patch.object(lp.Path, "read_text", lambda self, **kw: __import__("json").dumps(profile)),
                patch.dict("os.environ", {"VMS_LIVEPORTRAIT_ROOT": str(fake_root)}),
                patch.object(lp.subprocess, "run", fake_run),
            ):
                result = Path(lp.generate_from_expression_profile(
                    {"emotion": "joy", "identity": {"source_override": str(source)}}))

            self.assertTrue(seen["staged_exists"])
            self.assertNotIn("output", seen["runtime"].relative_to(tmp).parts)   # 暫存不在公開目錄
            self.assertFalse(seen["runtime"].exists())                           # 用完即刪
            self.assertTrue(result.is_file())                                    # 成品仍在 output
            self.assertIn("output", result.relative_to(tmp).parts)


if __name__ == "__main__":
    unittest.main()
