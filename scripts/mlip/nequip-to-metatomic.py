"""Turns a NequIP or Allegro model package (`.nequip.zip`) into a metatomic
artifact (TorchScript) that MDIR's metatomic adapter can load (D[metatomic-plugin],
docs/metatomic-ml1.md).

An example of bringing a model without an official metatomic export into
MDIR. The weights are not changed. The wrapper hands the energy network of
the package the pair list that the engine supplies (metatomic
`NeighborListOptions`, full and strict at the model's `r_max`) as NequIP's
edge vectors; the engine registers those vectors with autograd
(`register_autograd_neighbors`), so the forces and the virial are the
derivatives of the energy in the positions and the strain, as in NequIP's
own calculator, where they are the derivatives through its edge vectors.
Allegro is strictly local, so with `selected_atoms` the wrapper keeps only
the edges of the selected centers and returns their energies: an engine may
evaluate a large system in chunks of centers, with memory set by the chunk.

Tested with (2026-10-16): mir-group/Allegro-OAM-L:0.1 from Zenodo
(doi:10.5281/zenodo.16980200, file Allegro-OAM-L-0.1.nequip.zip, md5
0db7f9b3c3a62e74d78b3fcf2973c462, license CC-BY-4.0), packaged with
nequip 0.14.0, allegro 0.7.1, and torch 2.7.0; loaded and wrapped with torch
2.14.1, nequip 0.19.1, nequip-allegro 0.8.3 (MIT), metatomic-torch 0.1.19,
and metatensor-torch 0.10.8 (BSD-3-Clause). Neither the package nor the
artifact is part of MDIR.

Architecture: Musaelian et al., Nat. Commun. 14, 579 (2023),
doi:10.1038/s41467-023-36329-y. Interface: Bigi et al., J. Chem. Phys. 164,
064113 (2026), doi:10.1063/5.0304911.

usage: python nequip-to-metatomic.py PACKAGE.nequip.zip OUTPUT.pt
           [--dtype float64|float32] [--model sole_model]

`--dtype` is the type of the positions, cell, and pair vectors that the
engine hands over (metatomic's `ModelCapabilities.dtype`). The network keeps
its own type (float32 for Allegro-OAM-L); float64, the default, is what
NequIP's calculator hands it.
"""
import argparse
from typing import Dict, List, Optional

import ase.data
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)
from nequip.model.saved_models.load_utils import load_saved_model


class NequIPMetatomic(torch.nn.Module):
    """The energy network of a NequIP/Allegro model behind metatomic's model
    interface: energy only; forces and virial by the engine's autograd."""

    def __init__(self, energy_net: torch.nn.Module, type_of_z: torch.Tensor, r_max: float):
        super().__init__()
        self.net = energy_net
        self.register_buffer("type_of_z", type_of_z)
        self.nl = NeighborListOptions(cutoff=r_max, full_list=True, strict=True)

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.nl]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if "energy" not in outputs:
            return {}
        if outputs["energy"].sample_kind != "system":
            raise ValueError("only the total energy (sample_kind='system') is supported")
        energies: List[torch.Tensor] = []
        for s, system in enumerate(systems):
            nl = system.get_neighbor_list(self.nl)
            first = nl.samples.column("first_atom").to(torch.long)
            second = nl.samples.column("second_atom").to(torch.long)
            vec = nl.values.reshape(-1, 3)
            centers: Optional[torch.Tensor] = None
            if selected_atoms is not None:
                # Strictly local: a center's energy depends on its own edges.
                sel = selected_atoms.values
                centers = sel[sel[:, 0] == s][:, 1].to(torch.long)
                keep = torch.isin(first, centers)
                first, second, vec = first[keep], second[keep], vec[keep]
            n = system.positions.shape[0]
            data: Dict[str, torch.Tensor] = {
                "pos": system.positions,
                "edge_index": torch.stack([first, second], dim=0),
                "edge_vectors": vec,  # NequIP uses these instead of pos and shifts
                "atom_types": self.type_of_z[system.types],
                "batch": torch.zeros(n, dtype=torch.long, device=vec.device),
                "num_atoms": torch.tensor([n], dtype=torch.long, device=vec.device),
            }
            out = self.net(data)
            if centers is None:
                energies.append(out["total_energy"].reshape(1))
            else:
                atomic = out["atomic_energy"].reshape(-1)
                energies.append(atomic.index_select(0, centers).sum().reshape(1))
        e = torch.stack(energies).reshape(-1, 1).to(systems[0].positions.dtype)
        device = e.device
        block = TensorBlock(
            values=e,
            samples=Labels("system", torch.arange(len(systems), device=device).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]], device=device)),
        )
        return {"energy": TensorMap(Labels("_", torch.tensor([[0]], device=device)), [block])}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("package")
    p.add_argument("output")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--model", default="sole_model")
    a = p.parse_args()

    m = load_saved_model(a.package, compile_mode="eager", model_key=a.model)
    m.eval()
    r_max = float(m.metadata["r_max"])
    if "per_edge_type_cutoff" in m.metadata:
        raise SystemExit("per-edge-type cutoffs are not handled by this wrapper")
    type_names = m.metadata["type_names"].split(" ")
    zs = [ase.data.atomic_numbers[t] for t in type_names]
    type_of_z = torch.full((max(zs) + 1,), -1, dtype=torch.long)
    for i, z in enumerate(zs):
        type_of_z[z] = i
    net = m.model.func  # the energy network, inside NequIP's ForceStressOutput
    for q in net.parameters():
        q.requires_grad_(False)

    capabilities = ModelCapabilities(
        outputs={"energy": ModelOutput(unit="eV", sample_kind="system")},
        atomic_types=zs,
        interaction_range=r_max,
        length_unit="angstrom",
        supported_devices=["cuda", "cpu"],
        dtype=a.dtype,
    )
    metadata = ModelMetadata(
        name=f"{a.package.rsplit('/', 1)[-1]} (metatomic wrapper)",
        description="Energy network of a NequIP/Allegro package behind metatomic's "
        "interface, by MDIR's scripts/mlip/nequip-to-metatomic.py",
        references={"architecture": ["https://doi.org/10.1038/s41467-023-36329-y"]},
    )
    model = AtomisticModel(NequIPMetatomic(net, type_of_z, r_max).eval(), metadata, capabilities)
    model.save(a.output)
    print(f"saved {a.output}: {len(zs)} types, r_max {r_max} A, inputs {a.dtype}")


if __name__ == "__main__":
    main()
