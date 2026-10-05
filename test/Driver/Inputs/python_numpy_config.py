"""The real Python CMake configuration must reject missing or old NumPy."""
import pathlib
import subprocess
import sys
import venv

root = pathlib.Path(sys.argv[1]).resolve()
source = pathlib.Path(sys.argv[2]).resolve()
venv.EnvBuilder(with_pip=False).create(root / "env")
interpreter = root / "env/bin/python"
project = root / "project"
project.mkdir()
(project / "CMakeLists.txt").write_text(
    'cmake_minimum_required(VERSION 3.20)\nproject(numpy_dependency LANGUAGES CXX)\n'
    f'add_subdirectory("{source}" python)\n')
for name in ("missing", "old"):
    if name == "old":
        version = subprocess.check_output([str(interpreter), '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'], text=True).strip()
        numpy = root / f"env/lib/python{version}/site-packages/numpy"
        numpy.mkdir()
        # Control only the imported version; scientific routines are unused.
        (numpy / "__init__.py").write_text(
            "from types import SimpleNamespace\n__version__ = '1.22.4'\n"
            "lib = SimpleNamespace(NumpyVersion=lambda value: value)\n")
    result = subprocess.run(['cmake', '-S', str(project), '-B', str(root / name),
                              f'-DPython_EXECUTABLE={interpreter}'], capture_output=True, text=True)
    assert result.returncode != 0
    diagnostic = result.stdout + result.stderr
    assert 'requires NumPy >=1.23' in diagnostic, diagnostic
    assert ('1.22.4' in diagnostic if name == 'old' else 'numpy' in diagnostic)
print('missing and old NumPy refuse Python configuration with the minimum version')
