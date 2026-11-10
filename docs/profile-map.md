# The profile map

The profile map says, for each parallel region of a compiled program, what
the region computes: the op of the lowering it comes from, the set it runs
over, the potentials and terms its kernel holds, and the precision of its
arithmetic (D[profile-map], #840). The compiler writes it, so a profile is
read without guessing from the code of a region. This document covers the
host (the CPU target); the kernels of a device are to take the same records
once their lowering does (#840, after #795).

## Asking for it

| Where | How | Written |
|---|---|---|
| `mdir emit` | `mdir emit control.toml --stage=lowered --profile-map=map.json` | The map of the lowered program, beside what the stage prints. With `--stage=module` or `--stage=pipeline` the program is lowered as well, to make the map. |
| `mdir run` | `profile_map = "profile-map.json"` in `[output]` | The map of the program that the run compiles, once, before its steps. A run that is not continued keeps an earlier file of that name as `#<name>.<n>#`, as it keeps its other outputs (D149). |

The program of a run and the program that `mdir emit` lowers from the same
control file are the same, region for region, so either map serves a
profile of the run. A program on one thread has no parallel regions, and its
map lists none.

## The format

```json
{
  "format": 1,
  "target": "CPU",
  "precision": "MIXED",
  "threads": 2,
  "regions": [
    {
      "function": "mdir_run..omp_par.12",
      "parent": "mdir_run",
      "source": "pair_for",
      "set": "atoms",
      "potentials": ["energy"],
      "terms": [],
      "class": null,
      "precision": "f32+f64"
    }
  ]
}
```

A region is one `omp.parallel` of the lowered module, in the order of a
walk of its functions. Its keys:

| Key | What it is |
|---|---|
| `function` | The function that the translation to LLVM IR outlines the region into, as the symbols of the JIT and `mlir-translate` name it: `<parent>..omp_par` or `<parent>..omp_par.<n>`. |
| `parent` | The function the region is in: `mdir_run`, or a function of a template of the runtime (`mdrt.pme_spread_p64c32f32o4`). |
| `source` | The op of md_exec whose lowering made the loop, without its dialect: `pair_for`, `tuple_for`, `particle_for`, `scatter_add`, `refresh_neighbors`, `permute`, ...; or `template` for a loop of a template's function. `null` if the region has no record, which is a defect of the map. |
| `stage` | Where one op makes several loops, which one: for `refresh_neighbors`, `exclusion_filter` (the excluded pairs left out of a new list), `reference_positions` (the positions the list was built at), and `displacement_check` (whether a particle moved half the skin). |
| `set` | The set of the loop's rows: the tuple set of a `tuple_for` (`bonds`, `angles`, `dihedrals`, `pairs14`, `excluded`, `settles`, `shake1`, `anchors`, ...), else the particle set. |
| `template` | For a loop of a template, its function. |
| `potentials` | The potentials whose bodies the kernel holds: `energy`, the sum of the terms, for the forces of a step; `term_<name>` for the energy of one term, at the steps that write the energies. |
| `terms` | The terms the kernel holds, where the program names them: those of the potentials `term_<name>` (`term_lj` is `lj`), and the custom terms of the control file. A loop of the potential `energy` names no terms (see [What is left](#what-is-left)). |
| `class` | The class of the pairs (WW, SW, SS), where a lowering has one; `null` on the host. |
| `precision` | The floating-point types the arithmetic of the region computes in: `f32`, `f64`, `f32+f64`, or `""` for a region that only moves values (a permutation, a copy). |

## How it is made

The record of a region travels on locations, which every pass carries from
an op to the ops it is rewritten into, and which the translation to LLVM IR
drops, since it emits no debug information:

1. `md-inline` gives each op it copies from the body of a potential, or of a
   function that `md-differentiate` derived from one (`energy.forces`), a
   location named `potential: <name>`.
2. `convert-md-exec-to-loops` gives each op of md_exec, before its buffers
   lose the names of their sets, a `FusedLoc` whose metadata
   `mdir.profile` is its record (`profile::describe`,
   `include/mdir/Dialect/MDExec/ProfileMap.h`). Every op its lowering
   creates at that location inherits the record, the `scf.parallel` among
   them; a lowering that makes several loops refines it with a `stage`. The
   loops of the templates take the record of their function.
3. `convert-scf-to-openmp` makes each `scf.parallel` an `omp.parallel` at
   the same location, and so do the passes to the LLVM dialect.
4. The driver (`compiler::getProfileMap`, `lib/Compiler/ProfileMap.cpp`)
   copies the lowered module, puts a call of a mark with the region's index
   at the start of each `omp.parallel`, translates the copy, and reads the
   function each mark is in. The mark changes no order of outlining, so the
   names are those of the module's own translation.

The lowered module, its LLVM IR, and so the keys of the compile caches are
the same with the records as without them: the records are in locations
only, and the mark is in a copy. The names of the outlined functions are
LLVM's and are not changed.

## Reading a profile with it

`scripts/benchmarks/cpu-breakdown/prof.py` takes the part of each region
from its record: `pair_for` is "pairs"; a `tuple_for` is "excluded pairs"
over `excluded`, "constraints" over `settles` and `shake<n>`, "particles"
over `anchors`, and "bonded" over any other set; `particle_for` is
"particles"; `refresh_neighbors`, `spatial_order`, and `permute` are "list
builds"; a template is the part of the PME stage or the list build its
function names. `prof.py --classify CONTROL|MAP` prints the part and the
record of each region. Without a map `prof.py` gives the regions no part.

## What is left

- **The terms of the forces.** The builder writes the sum of the terms,
  `energy`, as one potential whose loops already join the terms (the
  Lennard-Jones and the Coulomb of the pairs in one kernel), and its ops
  carry no name of the term they come from. The loops of the forces
  therefore name the potential `energy` and no terms; the loops of the
  energies of one term name it. Naming the terms there needs the builder to
  locate the ops of each term, as it does for custom terms.
- **The kernels of a device.** MDExecToGPU is to attach the same records
  with `profile::describe`, and the map to pair them with the kernels and
  their launches (#840).
