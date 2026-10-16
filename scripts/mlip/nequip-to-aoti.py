"""Compiles a model package of the `nequip` package (`.nequip.zip`) into an
AOTInductor package (`.nequip.pt2`) for the `AOTI` backend of MDIR's plugin
of learned potentials (D[mlip-aoti], docs/metatomic-ml1.md, Section 11).

The package is the one that `nequip-compile --mode aotinductor --target ase`
writes, which the plugin loads as it is: the inputs `pos`, `edge_index`,
`atom_types`, `cell`, and `edge_cell_shift`, the outputs `atomic_energy`,
`total_energy`, `forces`, and `stress`, the forces and the stress being the
derivatives that the `nequip` package's `ForceStressOutput` takes inside the
compiled graph. This script runs the same export (the `nequip` package's
`aot_export_model`) and adds to the package's metadata what the `nequip`
package does not record and MDIR reads:

- `mdir::local_centers = "true"` for an Allegro model, which is strictly
  local (Musaelian et al., Nat. Commun. 14, 579 (2023),
  doi:10.1038/s41467-023-36329-y): the energy of a center depends only on its
  own edges, so the plugin may evaluate a system in chunks of centers. A
  NequIP model (message passing) is not declared local.
- `mdir::length_unit` and `mdir::energy_unit`, the units of the training data
  (`--units`, default Angstrom and eV, which the `nequip` package's ASE
  calculator assumes as well).

With `--arch`, the kernels are compiled for several kinds of device in one
package (AOTInductor's `emit_multi_arch_kernel`: SASS for each compute
capability listed and PTX for the device that compiles; this needs `nvcc` on
the path), and `mdir::archs` lists them; without it, for the device that
compiles alone, as `nequip-compile` does. The plugin refuses a package that
has no kernels for its device.

The package also holds host code, which AOTInductor compiles for the
compiling host's processor (`-march=native`); on a processor without its
instructions (AVX-512 on a host of AVX2) it stops with an illegal
instruction when it is loaded. This script compiles the host code for
x86-64-v3 (AVX2) by default (`--cpu`), so that a package runs on any host of
the last decade; the plugin refuses a package whose `AOTI_CPU_ISA` the host
lacks.

Tested with (2026-10-16): mir-group/Allegro-OAM-L:0.1 (Zenodo,
doi:10.5281/zenodo.16980200, CC-BY-4.0), torch 2.14.1, nequip 0.19.1,
nequip-allegro 0.8.3. Neither the model nor the package is part of MDIR.

usage: python nequip-to-aoti.py PACKAGE.nequip.zip OUTPUT.nequip.pt2
           [--model sole_model] [--modifiers enable_TritonContracter]
           [--arch 8.9 12.0] [--units angstrom eV]

Equivalent without MDIR's metadata (the plugin then evaluates the whole
system in one call and assumes Angstrom and eV):

    nequip-compile PACKAGE.nequip.zip OUTPUT.nequip.pt2 --mode aotinductor \\
        --device cuda --target ase [--modifiers enable_TritonContracter]
"""
import argparse
import os
import time

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("package")
    p.add_argument("output")
    p.add_argument("--model", default="sole_model")
    p.add_argument("--modifiers", nargs="*", default=[])
    p.add_argument("--arch", nargs="*", default=[],
                   help="compute capabilities to add SASS for, e.g. 8.9 12.0")
    p.add_argument("--units", nargs=2, default=["angstrom", "eV"],
                   metavar=("LENGTH", "ENERGY"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu", default="x86-64-v3",
                   help="the processor that the host code is compiled for (-march), "
                   "'native' for the compiling host's")
    a = p.parse_args()
    if a.cpu != "native":
        # x86-64-v3 is AVX2: the inductor's vector code and -march agree.
        os.environ.setdefault("ATEN_CPU_CAPABILITY",
                              "avx2" if a.cpu == "x86-64-v3" else "default")
    if not a.output.endswith(".nequip.pt2"):
        raise SystemExit("the output must end with .nequip.pt2")

    from nequip.data import AtomicDataDict
    from nequip.model.modify_utils import modify
    from nequip.model.saved_models.load_utils import load_saved_model
    from nequip.model.utils import _EAGER_MODEL_KEY
    from nequip.scripts._compile_utils import COMPILE_TARGET_DICT
    from nequip.scripts._workflow_utils import set_workflow_state
    from nequip.utils.aot import aot_export_model
    from nequip.utils.aoti_metadata import (
        NEQUIP_AOTI_INPUTS_KEY,
        NEQUIP_AOTI_OUTPUTS_KEY,
        serialize_aoti_keys,
    )
    from nequip.utils.global_state import get_latest_global_state, set_global_state

    # The steps of `nequip-compile --mode aotinductor --target ase`.
    set_workflow_state("compile")
    set_global_state(allow_tf32=False)
    device = torch.device(a.device)
    model, data = load_saved_model(a.package, _EAGER_MODEL_KEY, a.model,
                                   return_data_dict=True)
    model = modify(model, [{"modifier": m} for m in a.modifiers])
    metadata = model.metadata.copy()
    metadata.update(get_latest_global_state(only_metadata_related=True))
    metadata = {k: str(int(v)) if isinstance(v, bool) else v for k, v in metadata.items()}
    if "per_edge_type_cutoff" in metadata:
        raise SystemExit("per-edge-type cutoffs are not taken by MDIR's plugin")

    target = COMPILE_TARGET_DICT["ase"]
    data = AtomicDataDict.to_(target["data_settings"](data), device)
    batch_map = target["batch_map_settings"]({
        "graph": torch.export.Dim.STATIC,
        "node": torch.export.Dim("num_nodes", min=2, max=torch.inf),
        "edge": torch.export.Dim("num_edges", min=2, max=torch.inf),
    })

    # What MDIR reads beside the `nequip` package's own metadata.
    allegro = any(type(m).__name__ == "Allegro_Module" for m in model.modules())
    if allegro:
        metadata["mdir::local_centers"] = "true"
    metadata["mdir::length_unit"], metadata["mdir::energy_unit"] = a.units
    metadata[NEQUIP_AOTI_INPUTS_KEY] = serialize_aoti_keys(target["input"])
    metadata[NEQUIP_AOTI_OUTPUTS_KEY] = serialize_aoti_keys(target["output"])
    configs = {"aot_inductor.metadata": {k: str(v) for k, v in metadata.items()}}
    if a.cpu != "native":
        configs["cpp.march"] = a.cpu
    major, minor = torch.cuda.get_device_capability(device)
    if a.arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(a.arch)
        configs["aot_inductor.emit_multi_arch_kernel"] = True
        # AOTI_COMPUTE_CAPABILITY names the compiling device alone; MDIR's
        # plugin takes the package on every architecture listed here.
        archs = {f"{major}{minor}"} | {x.replace(".", "") for x in a.arch}
        configs["aot_inductor.metadata"]["mdir::archs"] = " ".join(sorted(archs, key=int))

    start = time.perf_counter()
    aot_export_model(model=model, device=device, input_fields=target["input"],
                     output_fields=target["output"], data=data, batch_map=batch_map,
                     output_path=a.output, inductor_configs=configs, seed=1)
    print(f"saved {a.output} in {time.perf_counter() - start:.0f} s: "
          f"r_max {metadata['r_max']}, model {metadata['model_dtype']}, "
          f"compiled on sm_{major}{minor}"
          + (f" with SASS for {' '.join(a.arch)}" if a.arch else "")
          + (", local centers" if allegro else ", not declared local")
          + (f", modifiers {' '.join(a.modifiers)}" if a.modifiers else ""))


if __name__ == "__main__":
    main()
