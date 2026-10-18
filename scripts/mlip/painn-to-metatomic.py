"""Turns a PaiNN checkpoint of SchNetPack 2 (a PyTorch Lightning checkpoint
whose `model.` entries are a `schnetpack.model.NeuralNetworkPotential` with a
`schnetpack.representation.PaiNN`, an `Atomwise` energy, and `AddOffsets`)
into a metatomic artifact (TorchScript) that MDIR's model term loads with
`backend = "METATOMIC"` (D270, docs/ml4-baselines.md).

An example of bringing a model without an official metatomic export into
MDIR, as `nequip-to-metatomic.py` is for the `nequip` package. The weights
are not changed. The checkpoint holds weights and no hyperparameters, so the
network is rebuilt from the arguments, whose defaults are those of the
nablaDFT benchmark's PaiNN (`config/model/painn.yaml` of nablaDFT 2.0): 6
interactions, 128 features, 100 Gaussians, a cosine cutoff at 5 A. A
checkpoint whose weights do not fit these shapes is refused.

The wrapper hands SchNetPack's PaiNN the pair list that the engine supplies
(metatomic `NeighborListOptions`, full and strict at the cutoff) as its pair
vectors `_Rij`, with `_idx_i` the first atom and `_idx_j` the second, the
convention of SchNetPack's own neighbor lists. The engine registers the
vectors with autograd, so the forces and the virial are the derivatives of
the energy in the positions and a strain, as in SchNetPack's calculator,
where they are derivatives through the same vectors. The energy is the sum
over atoms of the `Atomwise` network plus the per-atom mean of
`AddOffsets` (and its single-atom references where they are enabled), which
is what `NeuralNetworkPotential` computes with its postprocessing. PaiNN
passes messages, so an atom's energy depends on atoms up to 6 x 5 A away:
the artifact does not declare `mdir::local_centers`, and the engine
evaluates the whole system in one call.

Tested with (2026-10-18): the nablaDFT PaiNN of the training split
"train_large" (`painn_100k.ckpt`, SHA-256
6c81c40a99b2618ae4c0e5f96d441ece5c12bf47e1d8137cbd62883bcf77ef06), loaded
with torch 2.14.1, schnetpack 2.2.0, metatomic-torch 0.1.19, and
metatensor-torch 0.10.8. Neither the checkpoint nor the artifact is part of
MDIR; `fetch-ml4-artifacts.py` downloads and checks the checkpoint.

Architecture: Schuett, Unke, and Gastegger, ICML 2021, PMLR 139, 9377-9388.
Model: Khrabrov et al., NeurIPS 2024 Datasets and Benchmarks (nabla^2 DFT).
Interface: Bigi et al., J. Chem. Phys. 164, 064113 (2026),
doi:10.1063/5.0304911.

usage: python painn-to-metatomic.py CHECKPOINT.ckpt OUTPUT.pt
           [--dtype float64|float32] [--energy-unit Hartree]
           [--n-interactions 6] [--n-atom-basis 128] [--n-rbf 100]
           [--cutoff 5.0] [--types 1 6 7 8 9 16 17 35]

`--dtype` is the type of the positions, cell, and pair vectors that the
engine hands over; the network keeps its own (float32 for nablaDFT). The
default, float32, is what SchNetPack's calculator hands it.
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


def build_potential(n_interactions, n_atom_basis, n_rbf, cutoff):
    """SchNetPack's network of the nablaDFT configuration (painn.yaml)."""
    import schnetpack as spk
    import schnetpack.transform as trn

    representation = spk.representation.PaiNN(
        n_atom_basis=n_atom_basis,
        n_interactions=n_interactions,
        radial_basis=spk.nn.radial.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff),
        cutoff_fn=spk.nn.cutoff.CosineCutoff(cutoff),
    )
    return spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[
            spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="energy"),
            spk.atomistic.Forces(),
        ],
        postprocessors=[trn.AddOffsets("energy", add_mean=True)],
        do_postprocessing=True,
    )


def load_potential(path, n_interactions, n_atom_basis, n_rbf, cutoff):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = {k[len("model."):]: v for k, v in checkpoint["state_dict"].items()
             if k.startswith("model.")}
    potential = build_potential(n_interactions, n_atom_basis, n_rbf, cutoff)
    potential.load_state_dict(state, strict=True)
    return potential.eval()


class PaiNNMetatomic(torch.nn.Module):
    """SchNetPack's PaiNN behind metatomic's model interface: energy only;
    forces and virial by the engine's autograd."""

    def __init__(self, potential: torch.nn.Module, cutoff: float):
        super().__init__()
        self.representation = potential.representation
        self.outnet = potential.output_modules[0].outnet
        offsets = potential.postprocessors[0]
        self.register_buffer("mean", offsets.mean.reshape(-1)[:1].clone()
                             if offsets.add_mean else torch.zeros(1, dtype=torch.float64))
        self.register_buffer("atomref", offsets.atomref.clone() if offsets.add_atomrefs
                             else torch.zeros(1, dtype=torch.float64))
        self.add_atomrefs = bool(offsets.add_atomrefs)
        self.dtype = next(potential.parameters()).dtype
        self.nl = NeighborListOptions(cutoff=cutoff, full_list=True, strict=True)

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
            inputs: Dict[str, torch.Tensor] = {
                "_atomic_numbers": system.types.to(torch.long),
                "_Rij": nl.values.reshape(-1, 3).to(self.dtype),
                "_idx_i": nl.samples.column("first_atom").to(torch.long),
                "_idx_j": nl.samples.column("second_atom").to(torch.long),
                "_n_atoms": torch.tensor([n], dtype=torch.long, device=device),
                "_idx_m": torch.zeros(n, dtype=torch.long, device=device),
            }
            inputs = self.representation(inputs)
            per_atom = self.outnet(inputs["scalar_representation"]).reshape(-1).to(torch.float64)
            per_atom = per_atom + self.mean.to(torch.float64)
            if self.add_atomrefs:
                per_atom = per_atom + self.atomref.to(torch.float64)[system.types.to(torch.long)]
            if selected_atoms is not None:
                sel = selected_atoms.values
                centers = sel[sel[:, 0] == s][:, 1].to(torch.long)
                per_atom = per_atom.index_select(0, centers)
            energies.append(per_atom.sum().reshape(1))
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
    p.add_argument("checkpoint")
    p.add_argument("output")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float32")
    p.add_argument("--energy-unit", default="Hartree")
    p.add_argument("--n-interactions", type=int, default=6)
    p.add_argument("--n-atom-basis", type=int, default=128)
    p.add_argument("--n-rbf", type=int, default=100)
    p.add_argument("--cutoff", type=float, default=5.0)
    # The elements of nablaDFT's training set (H, C, N, O, F, S, Cl, Br);
    # the embedding takes any number up to 99, the training saw these.
    p.add_argument("--types", type=int, nargs="+", default=[1, 6, 7, 8, 9, 16, 17, 35])
    a = p.parse_args()

    potential = load_potential(a.checkpoint, a.n_interactions, a.n_atom_basis,
                               a.n_rbf, a.cutoff)
    for q in potential.parameters():
        q.requires_grad_(False)
    capabilities = ModelCapabilities(
        outputs={"energy": ModelOutput(unit=a.energy_unit, sample_kind="system")},
        atomic_types=a.types,
        interaction_range=a.n_interactions * a.cutoff,
        length_unit="angstrom",
        supported_devices=["cuda", "cpu"],
        dtype=a.dtype,
    )
    metadata = ModelMetadata(
        name=f"{a.checkpoint.rsplit('/', 1)[-1]} (metatomic wrapper)",
        description="SchNetPack's PaiNN behind metatomic's interface, by MDIR's "
        "scripts/mlip/painn-to-metatomic.py",
        references={"architecture": ["http://proceedings.mlr.press/v139/schutt21a.html"]},
    )
    model = AtomisticModel(PaiNNMetatomic(potential, a.cutoff).eval(), metadata, capabilities)
    model.save(a.output)
    print(f"saved {a.output}: {len(a.types)} types, cutoff {a.cutoff} A, "
          f"{a.n_interactions} interactions, inputs {a.dtype}, energy in {a.energy_unit}")


if __name__ == "__main__":
    main()
