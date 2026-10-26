#!/usr/bin/env bash
# Installs the wheels of a release as a user does and runs them (#624, the
# maintainer: the wheels are tested by pip, not only inspected):
#
#   scripts/release/check-wheel-install.sh [--work DIR] [--gpu INDEX]
#       [--examples DIR] [--dry-run] WHEEL...
#   scripts/release/check-wheel-install.sh --index URL --version VERSION
#       [--work DIR] [--gpu INDEX] [--examples DIR] [--dry-run] X.Y...
#
# The second form installs `mdir[cuda]==VERSION` from the package index URL
# (with PyPI as the extra index, for NVIDIA's wheels) for each CPython X.Y,
# as a user does after the upload to TestPyPI or PyPI (#650), and checks it
# the same way.
#
# For each wheel, mdir-VERSION-cpXY-cpXY-*.whl: the CPython X.Y of its tag
# from `uv python install` (UV, or uv on the path; its interpreters and cache
# under DIR), a fresh virtual environment in DIR, `pip install
# "WHEEL[cuda]"` from the file, then, from an empty directory and with
# CUDA_ROOT, CUDA_HOME, and CUDA_PATH unset, so that nothing of the source
# tree or of a toolkit is taken:
#   - `import mdir`, whose __version__ must be VERSION, from the
#     environment's site-packages;
#   - 20 steps of the ala3 example on the CPU (--examples DIR, examples/ala3
#     of this tree by default);
#   - with --gpu INDEX, for the first wheel, the same on that GPU
#     (CUDA_VISIBLE_DEVICES=INDEX); a caller that shares the device holds
#     its lock around this script.
# The extra `cuda` takes cuFFT and ptxas from NVIDIA's wheels, so pip needs
# the network (or an index of its own, PIP_INDEX_URL). DIR (default: a new
# directory under TMPDIR) holds the environments and the logs,
# DIR/<wheel>.log. --dry-run prints what each wheel takes and stops: no uv,
# no pip, no run. Exit 0 when every wheel passes.
set -euo pipefail

usage() {
  echo "usage: $0 [--work DIR] [--gpu INDEX] [--examples DIR] [--dry-run] WHEEL..." >&2
  echo "       $0 --index URL --version VERSION [--work DIR] [--gpu INDEX] [--examples DIR] [--dry-run] X.Y..." >&2
  exit 2
}
work=; gpu=; dry=0; index=; index_version=
examples=$(cd "$(dirname "$0")/../.." && pwd)/examples/ala3
wheels=()
while (( $# )); do
  case $1 in
    --work) (( $# >= 2 )) || usage; work=$2; shift ;;
    --gpu) (( $# >= 2 )) || usage; gpu=$2; shift ;;
    --examples) (( $# >= 2 )) || usage; examples=$2; shift ;;
    --index) (( $# >= 2 )) || usage; index=$2; shift ;;
    --version) (( $# >= 2 )) || usage; index_version=$2; shift ;;
    --dry-run) dry=1 ;;
    -*) usage ;;
    *) wheels+=("$1") ;;
  esac
  shift
done
(( ${#wheels[@]} )) || usage
# From an index, each argument is a CPython; it stands as the name of the
# wheel that pip will choose for it.
if [[ -n $index || -n $index_version ]]; then
  [[ -n $index && $index_version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
    { echo "--index takes --version VERSION, and --version takes --index" >&2; exit 2; }
  pythons=("${wheels[@]}"); wheels=()
  for python in "${pythons[@]}"; do
    [[ $python =~ ^3\.[0-9]+$ ]] || { echo "not a CPython X.Y: $python" >&2; exit 2; }
    wheels+=("mdir-$index_version-cp${python/./}-cp${python/./}-index.whl")
  done
fi
[[ -z $gpu || $gpu =~ ^[0-9]+$ ]] || { echo "--gpu takes the index of a GPU, not '$gpu'" >&2; exit 2; }

# The version and the CPython of each wheel, from its name (PEP 427).
pattern='^mdir-([0-9]+\.[0-9]+\.[0-9]+)-cp([0-9])([0-9]+)-cp[0-9]+-[a-z0-9_]+\.whl$'
for wheel in "${wheels[@]}"; do
  [[ $(basename "$wheel") =~ $pattern ]] ||
    { echo "not a wheel of mdir for CPython: $wheel" >&2; exit 2; }
done

if (( dry )); then
  first=1
  for wheel in "${wheels[@]}"; do
    [[ $(basename "$wheel") =~ $pattern ]]
    version=${BASH_REMATCH[1]} python=${BASH_REMATCH[2]}.${BASH_REMATCH[3]}
    runs="cpu"; (( first )) && [[ -n $gpu ]] && runs="cpu, gpu $gpu"
    from=file; [[ -n $index ]] && from="$index"
    echo "$(basename "$wheel"): CPython $python, version $version, from $from, runs: $runs"
    first=0
  done
  exit 0
fi

uv=${UV:-uv}
command -v "$uv" > /dev/null || { echo "uv is not on the path (or UV): it installs the interpreters" >&2; exit 1; }
[[ -f $examples/ala3.prmtop ]] || { echo "no ala3 example in $examples" >&2; exit 1; }
work=${work:-$(mktemp -d "${TMPDIR:-/tmp}/mdir-wheel-install.XXXXXX")}
mkdir -p "$work"
work=$(cd "$work" && pwd)
export UV_PYTHON_INSTALL_DIR=$work/python UV_CACHE_DIR=$work/cache/uv
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$work/cache/pip}

run_ala3() { # python target
  (cd "$work/empty" && env -u CUDA_ROOT -u CUDA_HOME -u CUDA_PATH -u PYTHONPATH \
    "$1" -I - "$examples" "$2" <<'EOF'
import sys
import mdir

d, target = sys.argv[1], sys.argv[2]
loaded = mdir.load_amber(f"{d}/ala3.prmtop", f"{d}/ala3.inpcrd")
system, state = loaded.make_system(), loaded.make_state()
system.electrostatics = mdir.Electrostatics.PME
system.rigid_hydrogen_bonds = system.rigid_water = True
integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
integrator.timestep = 0.002
ensemble.kind, ensemble.temperature = mdir.EnsembleKind.NVT, 300.0
execution, schedule = mdir.Execution(), mdir.Schedule()
execution.target = mdir.Target.GPU if target == "gpu" else mdir.Target.CPU
schedule.steps, schedule.energy_period = 20, 10
sim = mdir.Simulation(mdir.compile(system, state, integrator, ensemble,
                                   execution, schedule))
sim.run(20)
print(f"{target}: 20 steps of ala3, {len(sim.state().positions)} particles")
EOF
  )
}

status=0; first=1
mkdir -p "$work/empty"
for wheel in "${wheels[@]}"; do
  [[ $(basename "$wheel") =~ $pattern ]]
  version=${BASH_REMATCH[1]} python=${BASH_REMATCH[2]}.${BASH_REMATCH[3]}
  name=$(basename "$wheel" .whl)
  log=$work/$name.log
  env=$work/venv-cp${BASH_REMATCH[2]}${BASH_REMATCH[3]}
  if [[ -n $index ]]; then
    source=("mdir[cuda]==$version" --index-url "$index" --extra-index-url https://pypi.org/simple)
  else
    source=("$(cd "$(dirname "$wheel")" && pwd)/$(basename "$wheel")[cuda]")
  fi
  if ! {
    "$uv" python install -q "$python" &&
    interpreter=$("$uv" python find "$python") &&
    rm -rf "$env" && "$interpreter" -m venv "$env" &&
    "$env/bin/pip" install -q "${source[@]}" &&
    found=$(cd "$work/empty" && env -u PYTHONPATH "$env/bin/python" -I -c \
      'import mdir; print(mdir.__version__, mdir.__file__)') &&
    echo "import: $found" &&
    [[ ${found%% *} == "$version" && ${found#* } == "$env/"* ]] &&
    run_ala3 "$env/bin/python" cpu &&
    if (( first )) && [[ -n $gpu ]]; then
      CUDA_VISIBLE_DEVICES=$gpu run_ala3 "$env/bin/python" gpu
    fi
  } > "$log" 2>&1; then
    echo "$name: FAILED (log $log)" >&2
    tail -5 "$log" >&2
    status=1
  else
    echo "$name: CPython $python, $(grep -c ': 20 steps' "$log") run(s)"
  fi
  first=0
done
exit $status
