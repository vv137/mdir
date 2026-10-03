"""Exercise real doctor runs, including diagnostic retention and cleanup."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

executable = shutil.which(sys.argv[1])
mode = sys.argv[2]
with tempfile.TemporaryDirectory(prefix="doctor-test-") as temporary:
    root = Path(temporary)
    scratch = root / "temporary files"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch), TMP=str(scratch),
                       TEMP=str(scratch))
    # The test selects its own failure pipeline and devices.
    for key in ("MDIR_PIPELINE", "MDIR_REPRODUCER", "MDRT_DEVICE"):
        environment.pop(key, None)

    def run(arguments, status, **overrides):
        result = subprocess.run([executable, "doctor", *arguments], cwd=root,
                                env=dict(environment, **overrides), text=True,
                                capture_output=True, timeout=300)
        assert result.returncode == status, result.stdout + result.stderr
        assert set(root.iterdir()) == {scratch}, list(root.iterdir())
        return result.stdout + result.stderr

    def retained():
        directories = list(scratch.iterdir())
        assert len(directories) == 1, directories
        directory = directories[0]
        assert (directory / "atoms.pdb").is_file()
        return directory

    if mode == "cpu":
        output = run(["--target=cpu"], 0, CUDA_VISIBLE_DEVICES="-1")
        assert "PASS cpu:" in output and "CUDA driver API:" not in output, output
        assert "all requested checks passed" in output, output
        assert not list(scratch.iterdir())
        if "targets: cpu\n" in output:
            output = run([], 0)
            assert "SKIP gpu: this build has no CUDA target" in output, output
            assert not list(scratch.iterdir())
            output = run(["--target=gpu"], 1)
            assert "FAIL gpu: this build has no CUDA target" in output, output
            assert not list(scratch.iterdir())
        output = run(["--target=unknown"], 1)
        assert "expects --target=all, cpu, or gpu" in output, output
        assert not list(scratch.iterdir())
        output = run(["--target=cpu"], 1,
                     MDIR_PIPELINE="convert-md-exec-to-gpu",
                     MDIR_REPRODUCER=str(root / "must-not-write.mlir"))
        directory = retained()
        assert "FAIL cpu: run exited with status" in output, output
        assert str(directory) in output and "rerun with" in output, output
        assert (directory / "cpu.toml").is_file()
        assert (directory / "cpu.out").is_file()
        assert "not in the storage form" in (directory / "cpu.err").read_text()
        assert (directory / "cpu-reproducer.mlir").is_file()
    elif mode == "gpu":
        output = run([], 0)
        for expected in ("PASS cpu:", "CUDA driver API:", "CUDA device 0:",
                         "compute capability", "[selected]", "PASS gpu:"):
            assert expected in output, output
        assert not list(scratch.iterdir())
        output = run([], 1, CUDA_VISIBLE_DEVICES="-1")
        assert "PASS cpu:" in output and "SKIP gpu:" not in output, output
        assert "FAIL CUDA" in output and "--target=cpu" in output, output
        shutil.rmtree(retained())
        output = run(["--target=gpu"], 1, MDRT_DEVICE="invalid")
        assert "FAIL CUDA device: MDRT_DEVICE" in output, output
        shutil.rmtree(retained())
        # The CPU child fails, but the GPU is still probed and attempted.
        output = run([], 1, MDIR_PIPELINE="convert-md-exec-to-gpu")
        assert "FAIL cpu:" in output and "FAIL gpu:" in output, output
        directory = retained()
        assert (directory / "gpu.err").is_file()
    else:
        raise AssertionError(mode)
print(f"doctor {mode}: success, failures, and temporary files checked")
