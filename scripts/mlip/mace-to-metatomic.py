r"""Turns a MACE model file of mace-torch (`.model`, a pickled `MACE` or
`ScaleShiftMACE` with one head) into a metatomic artifact (TorchScript) that
MDIR's model term loads with `backend = "METATOMIC"` (D270,
docs/ml4-baselines.md).

An example of bringing a model into MDIR without changing it, as
`nequip-to-metatomic.py` is for the `nequip` package. The weights are not
changed, and the model runs in float64, the type in which these files are
stored (and in which MACE-OFF23 was trained and is evaluated by mace-torch's
`mace_off()`). metatrain's experimental MACE architecture
(metatrain 2026.4.1, `metatrain.experimental.mace`) also exports such files
to metatomic, by a training of zero epochs that moves the reference
energies and the scale into its own heads; this wrapper calls the model as
mace-torch's calculator does instead, so that the network evaluated is the
file's.

The wrapper builds mace-torch's input from the engine's pair list (metatomic
`NeighborListOptions`, full and strict at the model's `r_max`): the edges
`edge_index` = (first atom, second atom), the shifts `shifts` = $\mathbf
s\,\mathsf H$ from the list's cell shifts and the system's cell, and the
one-hot of the elements. MACE forms its edge vectors itself, $\mathbf x_j -
\mathbf x_i + \mathbf s\,\mathsf H$, from the positions and the cell of the
system, which the engine differentiates in the positions and a strain: the
forces and the virial are those derivatives, as in mace-torch's calculator
(where the virial is the derivative in a symmetric displacement of the same
positions and cell). The energy is the sum of MACE's `node_energy`: the
reference energies of the elements, the interaction energies with their
scale and shift. MACE passes messages ($L$ layers of cutoff $r_c$), so an
atom's energy depends on atoms up to $L\,r_c$ away: the artifact does not
declare `mdir::local_centers`, and the engine evaluates the whole system in
one call.

Tested with (2026-10-18): MACE-OFF23 medium (`MACE-OFF23_medium.model`,
SHA-256 4842c52ad210d6e1f84d6cf1ffa70fae25a7e0d755ed55cf223f43913f587db7)
and MACE-MP-0 medium (`2023-12-03-mace-128-L1_epoch-199.model`, SHA-256
01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845), loaded
with torch 2.14.1, mace-torch 0.3.16, e3nn 0.4.4, metatomic-torch 0.1.19,
and metatensor-torch 0.10.8. Neither the model file nor the artifact is
part of MDIR; `fetch-ml4-artifacts.py` downloads and checks the file.

Architecture: Batatia et al., NeurIPS 35 (2022), arXiv:2206.07697. Interface:
Bigi et al., J. Chem. Phys. 164, 064113 (2026), doi:10.1063/5.0304911.

usage: python mace-to-metatomic.py MODEL.model OUTPUT.pt

The network, the positions, the cell, and the pair vectors that the engine
hands over are float64: MACE computes in the type of its weights.
"""
import argparse
from typing import Dict, List, Optional

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


class MACEMetatomic(torch.nn.Module):
    """A mace-torch model behind metatomic's model interface: energy only;
    forces and virial by the engine's autograd."""

    def __init__(self, mace: torch.nn.Module, r_max: float):
        super().__init__()
        self.mace = mace
        zs = mace.atomic_numbers.tolist()
        index_of_z = torch.full((max(zs) + 1,), -1, dtype=torch.long)
        for i, z in enumerate(zs):
            index_of_z[z] = i
        self.register_buffer("index_of_z", index_of_z)
        self.n_types = len(zs)
        self.dtype = next(mace.parameters()).dtype
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
            n = system.positions.shape[0]
            device = system.positions.device
            first = nl.samples.column("first_atom").to(torch.long)
            second = nl.samples.column("second_atom").to(torch.long)
            unit_shifts = torch.stack(
                [nl.samples.column("cell_shift_a"), nl.samples.column("cell_shift_b"),
                 nl.samples.column("cell_shift_c")], dim=1).to(self.dtype)
            cell = system.cell.to(self.dtype)
            species = self.index_of_z[system.types.to(torch.long)]
            data: Dict[str, torch.Tensor] = {
                "positions": system.positions.to(self.dtype),
                "cell": cell.reshape(1, 3, 3),
                "edge_index": torch.stack([first, second], dim=0),
                "unit_shifts": unit_shifts,
                "shifts": unit_shifts @ cell,
                "node_attrs": torch.nn.functional.one_hot(
                    species, self.n_types).to(self.dtype),
                "batch": torch.zeros(n, dtype=torch.long, device=device),
                "ptr": torch.tensor([0, n], dtype=torch.long, device=device),
            }
            out = self.mace(data, training=False, compute_force=False)
            node_energy = out["node_energy"]
            assert node_energy is not None
            node_energy = node_energy.reshape(-1).to(torch.float64)
            if selected_atoms is not None:
                sel = selected_atoms.values
                centers = sel[sel[:, 0] == s][:, 1].to(torch.long)
                node_energy = node_energy.index_select(0, centers)
            energies.append(node_energy.sum().reshape(1))
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
    p.add_argument("model")
    p.add_argument("output")
    a = p.parse_args()

    mace = torch.load(a.model, map_location="cpu", weights_only=False)
    if getattr(mace, "heads", None) is not None and len(mace.heads) > 1:
        raise SystemExit(f"a model of several heads ({mace.heads}) is not handled")
    mace = mace.to(torch.float64).eval()
    for q in mace.parameters():
        q.requires_grad_(False)
    r_max = float(mace.r_max)
    layers = int(mace.num_interactions)
    zs = mace.atomic_numbers.tolist()

    capabilities = ModelCapabilities(
        outputs={"energy": ModelOutput(unit="eV", sample_kind="system")},
        atomic_types=zs,
        interaction_range=layers * r_max,
        length_unit="angstrom",
        supported_devices=["cuda", "cpu"],
        dtype="float64",
    )
    metadata = ModelMetadata(
        name=f"{a.model.rsplit('/', 1)[-1]} (metatomic wrapper)",
        description="A mace-torch model behind metatomic's interface, by MDIR's "
        "scripts/mlip/mace-to-metatomic.py",
        references={"architecture": ["https://arxiv.org/abs/2206.07697"]},
    )
    wrapper = MACEMetatomic(mace, r_max).eval()
    # e3nn's modules declare how each is compiled (some are traced, not
    # scripted); e3nn compiles the model as mace-torch's LAMMPS export does.
    from e3nn.util import jit as e3nn_jit

    wrapper.mace = e3nn_jit.compile(mace)
    model = AtomisticModel(wrapper, metadata, capabilities)
    model.save(a.output)
    print(f"saved {a.output}: {len(zs)} types, r_max {r_max} A, {layers} layers, "
          "float64")


if __name__ == "__main__":
    main()
