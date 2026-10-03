"""Bad NN configuration must fail before model work or output creation.

Feature/model readers are stubbed; no course data or saved weights are needed.
"""
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import predict_v22 as cli


class PredictionPreflightTests(unittest.TestCase):
    def invoke(self, output, interpreter, probe_error=None):
        config = {"inputs": {k: {"sha256": "ok"} for k in ("test", "sample")},
                  "feature_frame_sha256": "features"}
        raw = MagicMock()
        raw.reset_index.return_value = raw
        raw.__len__.return_value = 47073
        raw.__contains__.return_value = False
        sample = MagicMock()
        sample.__iter__.side_effect = lambda: iter(["編號", "總價元"])
        argv = ["predict_v22.py", "--test", "test.xlsx", "--sample", "sample.csv",
                "--output-dir", str(output), "--nn-python", str(interpreter)]
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.object(cli, "read", side_effect=[config, {"files": []}]))
            stack.enter_context(patch.object(cli, "digest", return_value="ok"))
            stack.enter_context(patch.object(cli.pd, "read_excel", return_value=raw))
            stack.enter_context(patch.object(cli.pd, "read_csv", return_value=sample))
            stack.enter_context(patch.object(cli.pd, "read_pickle"))
            stack.enter_context(patch.object(cli, "ids", return_value=["1"]))
            stack.enter_context(patch.object(cli, "build_features"))
            stack.enter_context(patch.object(cli, "frame_sha", return_value="features"))
            tree = stack.enter_context(patch.object(cli, "predict_trees",
                side_effect=AssertionError("Expensive model work was reached")))
            probe = stack.enter_context(patch.object(cli.subprocess, "run", side_effect=probe_error))
            try:
                cli.main()
            finally:
                self.assertFalse(tree.called, "NN configuration was checked after tree prediction")
                self.assertFalse(output.exists(), "Bad configuration left an output directory")
        return probe

    def test_missing_interpreter_rejected_before_model_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "nn-python"):
                self.invoke(Path(tmp) / "run", Path(tmp) / "missing.exe")

    def test_dependency_or_device_failure_rejected_before_model_work(self):
        for reason in ("No module named 'torch'", "CUDA unavailable"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                error = subprocess.CalledProcessError(1, [sys.executable], stderr=reason)
                with self.assertRaisesRegex(RuntimeError, reason):
                    self.invoke(Path(tmp) / "run", Path(sys.executable), error)

    def test_unlaunchable_interpreter_or_timeout_leaves_no_output(self):
        for error in (OSError("Not a Python executable"),
                      subprocess.TimeoutExpired([sys.executable], 60)):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(RuntimeError, "Cannot check --nn-python"):
                    self.invoke(Path(tmp) / "run", Path(sys.executable), error)

    def test_existing_output_is_preserved_without_launching_nn(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            output.mkdir()
            existing = output / "receipt.json"
            existing.write_bytes(b"existing result")
            argv = ["predict_v22.py", "--test", "test.xlsx", "--sample", "sample.csv",
                    "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), patch.object(cli, "read") as read, \
                 patch.object(cli.subprocess, "run") as probe:
                with self.assertRaises(FileExistsError):
                    cli.main()
                read.assert_not_called()
                probe.assert_not_called()
            self.assertEqual(existing.read_bytes(), b"existing result")

    def test_unexpected_probe_response_is_rejected(self):
        for body in ('not json', '[]', '{"status":"ready","device":"cpu"}'):
            with self.subTest(body=body), patch.object(cli.subprocess, "run",
                    return_value=subprocess.CompletedProcess([], 0, stdout=body)):
                with self.assertRaisesRegex(RuntimeError, "invalid result"):
                    cli.check_nn_environment(Path(sys.executable), "cuda:0")

    def test_symlink_interpreter_path_is_not_replaced_with_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            interpreter = Path(tmp) / "venv-python.exe"
            try:
                interpreter.symlink_to(sys.executable)
            except OSError as error:
                self.skipTest(f"This host cannot create file symlinks: {error}")
            response = subprocess.CompletedProcess([], 0,
                stdout='{"status":"ready","device":"cuda:0"}')
            with patch.object(cli.subprocess, "run", return_value=response) as probe:
                selected, _ = cli.check_nn_environment(interpreter, "cuda:0")
            self.assertEqual(selected, interpreter.absolute())
            self.assertEqual(probe.call_args.args[0][0], str(interpreter.absolute()))


if __name__ == "__main__":
    unittest.main()
