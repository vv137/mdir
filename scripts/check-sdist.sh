#!/usr/bin/env bash
# The Python package from its sdist, as a user builds it (#624):
#
#   scripts/check-sdist.sh WORK_DIR [CONFIG_SETTING...]
#
# builds the sdist of the tree, extracts it, builds the wheel from it (the
# sdist leaves test/ out, and pyproject.toml builds with MDIR_BUILD_TESTS=OFF),
# installs the wheel into a new virtual environment, and runs 20 steps of the
# ala3 example on the CPU from there. CONFIG_SETTING are passed to pip as
# `-C`, for example cmake.define.LLVM_DIR=<llvm>/lib/cmake/llvm. The tree is
# not changed; WORK_DIR holds the environments, the sdist, and the wheel.
# PYTHON names the interpreter (python3 by default). A wheel that auditwheel
# has not repaired links HDF5 from where it was built: HDF5_LIBRARY_DIR names
# that directory for the run (#634).
set -euo pipefail
work=${1:?usage: $0 WORK_DIR [CONFIG_SETTING...]}
shift
source=$(cd "$(dirname "$0")/.." && pwd)
python=${PYTHON:-python3}
mkdir -p "$work"
work=$(cd "$work" && pwd)
settings=()
for setting in "$@"; do settings+=(-C "$setting"); done

log() { printf '[check-sdist %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "the tools of the build"
"$python" -m venv "$work/build-env"
"$work/build-env/bin/pip" install -q build

log "the sdist"
rm -rf "$work/dist" "$work/sdist"
"$work/build-env/bin/python" -m build --sdist --outdir "$work/dist" "$source" > "$work/sdist.log"
sdist=$(ls "$work"/dist/mdir-*.tar.gz)
mkdir -p "$work/sdist"
tar -xzf "$sdist" -C "$work/sdist"
tree=$(ls -d "$work"/sdist/mdir-*)
[[ ! -e "$tree/test" ]] || { echo "the sdist holds test/" >&2; exit 1; }
log "$(basename "$sdist"), without test/"

log "the wheel, from the sdist"
"$work/build-env/bin/pip" wheel -q --no-deps -w "$work/dist" "${settings[@]}" "$tree" > "$work/wheel.log" 2>&1 || {
  tail -40 "$work/wheel.log" >&2; exit 1; }
wheel=$(ls "$work"/dist/mdir-*.whl)
log "$(basename "$wheel")"

log "a clean environment, the wheel installed"
rm -rf "$work/env"
"$python" -m venv "$work/env"
"$work/env/bin/pip" install -q "$wheel"
cd "$work"
LD_LIBRARY_PATH="${HDF5_LIBRARY_DIR:+$HDF5_LIBRARY_DIR:}${LD_LIBRARY_PATH:-}" \
"$work/env/bin/python" - "$source/examples/ala3" <<'EOF'
import sys
import mdir

d = sys.argv[1]
loaded = mdir.load_amber(f"{d}/ala3.prmtop", f"{d}/ala3.inpcrd")
system, state = loaded.make_system(), loaded.make_state()
system.electrostatics = mdir.Electrostatics.PME
system.rigid_hydrogen_bonds = system.rigid_water = True
integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
integrator.timestep = 0.002
ensemble.kind, ensemble.temperature = mdir.EnsembleKind.NVT, 300.0
execution, schedule = mdir.Execution(), mdir.Schedule()
execution.target = mdir.Target.CPU
schedule.steps = 20
schedule.energy_period = 10
sim = mdir.Simulation(mdir.compile(system, state, integrator, ensemble,
                                   execution, schedule))
sim.run(20)
print(f"check-sdist: mdir {mdir.__version__} from {mdir.__file__}")
print(f"check-sdist: 20 steps of ala3 on the CPU, {len(sim.state().positions)} particles")
EOF
log "ok"
