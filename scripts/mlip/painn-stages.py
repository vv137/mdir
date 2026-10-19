#!/usr/bin/env python3
"""The stages of a PaiNN of SchNetPack 2 as batched bodies for MDIR's neural
stages (#377 step 4, D[ml4-painn-stages], docs/ml4-ir.md, Section 8).

PaiNN (Schuett, Unke, and Gastegger, ICML 2021, PMLR 139, 9377-9388) in the
configuration of nablaDFT's benchmark (Khrabrov et al., NeurIPS 2024 Datasets
and Benchmarks; `painn_100k.ckpt` of D286) is written as stages of one
element each, batched over rows:

  embed(z)                -> h            the species' embedding; vectors 0
  ctx_k(h)                -> x            per atom: Dense(silu(Dense(q)))
  edge_k(r, d, x_j, h_j)  -> a            per directed pair (i, j)
  node_k(h, a)            -> h'           the update and the mixing block
  readout(h)              -> e            the atom's energy, kJ/mol

for the layers k = 1..L. h is `irreps<F x 0e + F x 1o, ir_mul>`: the F
scalars q, then the F x-components of the vectors mu, the F y-, and the F
z-components (SchNetPack's [3, F] per atom). x has 3F scalars. r and d are
MDIR's: the distance in nm and d = x_i - x_j; SchNetPack's pair vector is
r_ij = x_j - x_i = -d, in Angstrom. The conversions to Angstrom and from
Hartree to kJ/mol are inside the bodies, so that they are differentiated
with them.

Subcommands, each in its own environment:

  weights CHECKPOINT OUT.npz        the environment of ml4-requirements.lock
      Reads the checkpoint with SchNetPack (as painn-to-metatomic.py does)
      and writes its weights and, for the check, each stage of SchNetPack's
      own modules on random rows (outputs and products with random
      cotangents).
  weights --synthetic F L OUT.npz   any Python with NumPy
      A model of the same structure with F features, L layers, and random
      weights from a fixed seed, for MDIR's tests.
  check OUT.npz [REFERENCE.npz ...] JAX
      Each JAX stage against SchNetPack's modules (the rows of `weights`),
      and the energy, forces, and virial composed from the JAX stages, with
      MDIR's reverse rules (docs/ml4-ir.md, Section 4.1) applied to the
      stages' products, against references of water-box.py
      (`--kind painn --painn-dtype float64`: SchNetPack's calculator with
      its Strain input, the same weights in float64), within 1e-6 of the
      energy, 3e-5 of the largest force, and 1e-5 of the largest virial
      component. The stages compute in float32 and sum in float64; the
      package's own float32 evaluation is 3.4e-5 of the largest force from
      its float64 one on the 81-atom lattice, the stages 5.4e-6.
  export OUT.npz MODEL.mlir --rows T [--stablehlo-opt PATH]   JAX
      jax.export of each stage's forward and jax.vjp with T rows to
      StableHLO, legalized to linalg on tensors by StableHLO's
      `stablehlo-legalize-to-linalg`, and the md.stage declarations and the
      md.potential of the model's structure, in one MLIR module.

Neither the weights nor the exported module is part of MDIR.
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

HARTREE_KJ_PER_MOL = 2625.4996394798254  # CODATA 2018, as ASE's units
ANGSTROM_PER_NM = 10.0


# ---------------------------------------------------------------------------
# weights: from SchNetPack's checkpoint, or synthetic
# ---------------------------------------------------------------------------

def weights_from_checkpoint(path):
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location(
        "painn_to_metatomic", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "painn-to-metatomic.py"))
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    net = wrapper.load_potential(path, 6, 128, 100, 5.0)
    rep = net.representation
    offsets = net.postprocessors[0]
    if offsets.add_atomrefs:
        raise SystemExit("atom references are not supported (AddOffsets.add_atomrefs)")
    w = {
        "F": np.array(rep.n_atom_basis), "L": np.array(rep.n_interactions),
        "cutoff": rep.cutoff_fn.cutoff.detach().numpy().astype(np.float32).reshape(()),
        "rbf_offsets": rep.radial_basis.offsets.detach().numpy(),
        "rbf_widths": rep.radial_basis.widths.detach().numpy(),
        "embedding": rep.embedding.weight.detach().numpy(),
        "filter_w": rep.filter_net.weight.detach().numpy(),
        "filter_b": rep.filter_net.bias.detach().numpy(),
        "out_w1": net.output_modules[0].outnet[0].weight.detach().numpy(),
        "out_b1": net.output_modules[0].outnet[0].bias.detach().numpy(),
        "out_w2": net.output_modules[0].outnet[1].weight.detach().numpy(),
        "out_b2": net.output_modules[0].outnet[1].bias.detach().numpy(),
        "mean": np.array(float(offsets.mean.reshape(-1)[0]) if offsets.add_mean else 0.0),
        "epsilon": np.array(rep.mixing[0].epsilon, dtype=np.float32),
    }
    for k in range(rep.n_interactions):
        c = rep.interactions[k].interatomic_context_net
        m = rep.mixing[k]
        w[f"ctx{k}_w1"] = c[0].weight.detach().numpy()
        w[f"ctx{k}_b1"] = c[0].bias.detach().numpy()
        w[f"ctx{k}_w2"] = c[1].weight.detach().numpy()
        w[f"ctx{k}_b2"] = c[1].bias.detach().numpy()
        w[f"mix{k}_wmu"] = m.mu_channel_mix.weight.detach().numpy()
        w[f"mix{k}_w1"] = m.intraatomic_context_net[0].weight.detach().numpy()
        w[f"mix{k}_b1"] = m.intraatomic_context_net[0].bias.detach().numpy()
        w[f"mix{k}_w2"] = m.intraatomic_context_net[1].weight.detach().numpy()
        w[f"mix{k}_b2"] = m.intraatomic_context_net[1].bias.detach().numpy()
    w.update(torch_stage_rows(net, w))
    return w


def torch_stage_rows(net, w, rows=64, seed=7):
    """Each stage of SchNetPack's own modules on random rows, and its
    products with random cotangents (torch.autograd), for `check`."""
    import torch
    rep = net.representation
    F = int(w["F"])
    rng = np.random.default_rng(seed)
    t = lambda a: torch.tensor(a, dtype=torch.float32, requires_grad=True)
    out = {}

    def record(name, inputs, outputs):
        cot = [rng.normal(size=o.shape).astype(np.float32) for o in outputs]
        grads = torch.autograd.grad(outputs, inputs, [torch.tensor(c) for c in cot],
                                    allow_unused=True)
        for n, i in enumerate(inputs):
            out[f"ref_{name}_in{n}"] = i.detach().numpy()
            g = grads[n]
            out[f"ref_{name}_vjp{n}"] = np.zeros(i.shape, np.float32) if g is None else g.numpy()
        out[f"ref_{name}_out0"] = outputs[0].detach().numpy()
        out[f"ref_{name}_cot0"] = cot[0]

    h = rng.normal(size=(rows, 4 * F)).astype(np.float32)
    a = rng.normal(size=(rows, 4 * F)).astype(np.float32) * 0.3
    x = rng.normal(size=(rows, 3 * F)).astype(np.float32)
    r = rng.uniform(0.08, 0.55, size=rows)  # nm; some beyond the cutoff
    u = rng.normal(size=(rows, 3))
    d = u / np.linalg.norm(u, axis=1, keepdims=True) * r[:, None]
    for k in (0, rep.n_interactions - 1):
        # The context net of interaction k, per atom.
        hq = t(h)
        xk = rep.interactions[k].interatomic_context_net(hq[:, :F])
        record(f"ctx{k}", [hq], [xk])
        # The edge: SchNetPack's radial basis, cutoff, and filter network.
        rt, dt, xt, ht = t(r.astype(np.float32)), t(d.astype(np.float32)), t(x), t(h)
        dist = (rt * ANGSTROM_PER_NM)[:, None]
        rij = -dt * ANGSTROM_PER_NM
        phi = rep.radial_basis(dist)
        fcut = rep.cutoff_fn(dist)
        filters = (rep.filter_net(phi) * fcut[..., None])[:, 0, :]  # as PaiNN.forward
        W = filters[:, 3 * F * k:3 * F * (k + 1)]
        m = W * xt
        dq, dmuR, dmumu = torch.split(m, F, dim=-1)
        mu_j = ht[:, F:].reshape(rows, 3, F)
        dmu = dmuR[:, None, :] * (rij / dist)[..., None] + dmumu[:, None, :] * mu_j
        record(f"edge{k}", [rt, dt, xt, ht], [torch.cat([dq, dmu.reshape(rows, 3 * F)], -1)])
        # The node: the update, then SchNetPack's mixing block.
        ht, at = t(h), t(a)
        q = (ht[:, :F] + at[:, :F])[:, None, :]
        mu = (ht[:, F:] + at[:, F:]).reshape(rows, 3, F)
        q2, mu2 = rep.mixing[k](q, mu)
        record(f"node{k}", [ht, at], [torch.cat([q2[:, 0], mu2.reshape(rows, 3 * F)], -1)])
    # The readout: Atomwise's network, the mean of AddOffsets, Hartree.
    ht = t(h)
    e = (net.output_modules[0].outnet(ht[:, :F])[:, 0] + float(w["mean"])) * HARTREE_KJ_PER_MOL
    record("readout", [ht], [e])
    return out


def synthetic_weights(F, L, seed=3):
    """A PaiNN of the same structure with random weights (for tests)."""
    rng = np.random.default_rng(seed)
    n_rbf = 8
    cutoff = 5.0
    g = lambda *s: (rng.normal(size=s) / np.sqrt(s[-1])).astype(np.float32)
    offsets = np.linspace(0, cutoff, n_rbf).astype(np.float32)
    w = {
        "F": np.array(F), "L": np.array(L), "cutoff": np.array(cutoff, np.float32),
        "rbf_offsets": offsets,
        "rbf_widths": np.full(n_rbf, offsets[1] - offsets[0], np.float32),
        "embedding": g(100, F),
        "filter_w": g(3 * F * L, n_rbf), "filter_b": g(3 * F * L) * 0.1,
        "out_w1": g(max(F // 2, 1), F), "out_b1": g(max(F // 2, 1)) * 0.1,
        "out_w2": g(1, max(F // 2, 1)), "out_b2": g(1) * 0.1,
        "mean": np.array(-0.5), "epsilon": np.array(1e-8, np.float32),
    }
    for k in range(L):
        w[f"ctx{k}_w1"], w[f"ctx{k}_b1"] = g(F, F), g(F) * 0.1
        w[f"ctx{k}_w2"], w[f"ctx{k}_b2"] = g(3 * F, F), g(3 * F) * 0.1
        w[f"mix{k}_wmu"] = g(2 * F, F)
        w[f"mix{k}_w1"], w[f"mix{k}_b1"] = g(F, 2 * F), g(F) * 0.1
        w[f"mix{k}_w2"], w[f"mix{k}_b2"] = g(3 * F, F), g(3 * F) * 0.1
    return w


# ---------------------------------------------------------------------------
# The stages in JAX
# ---------------------------------------------------------------------------

class Stages:
    """PaiNN's stages, batched over rows, in float32 (the checkpoint's type)."""

    def __init__(self, w):
        import jax.numpy as jnp
        self.jnp = jnp
        self.w = {k: (jnp.asarray(v) if v.dtype != object and v.ndim >= 1 else v)
                  for k, v in w.items() if not k.startswith("ref_")}
        self.F = int(w["F"])
        self.L = int(w["L"])
        self.cutoff = float(w["cutoff"])
        self.mean = float(w["mean"])
        self.eps = float(w["epsilon"])

    def linear(self, x, wname, bname=None, act=False):
        import jax
        y = x @ self.w[wname].T
        if bname is not None:
            y = y + self.w[bname]
        return jax.nn.silu(y) if act else y

    def embed(self, z):
        jnp = self.jnp
        q = self.w["embedding"][z]
        return jnp.concatenate([q, jnp.zeros((z.shape[0], 3 * self.F), q.dtype)], -1)

    def ctx(self, k, h):
        F = self.F
        y = self.linear(h[:, :F], f"ctx{k}_w1", f"ctx{k}_b1", act=True)
        return self.linear(y, f"ctx{k}_w2", f"ctx{k}_b2")

    def edge(self, k, r, d, xj, hj):
        jnp = self.jnp
        F = self.F
        dist = r * ANGSTROM_PER_NM
        rij = -d * ANGSTROM_PER_NM
        coeff = -0.5 / self.w["rbf_widths"] ** 2
        phi = jnp.exp(coeff * (dist[:, None] - self.w["rbf_offsets"]) ** 2)
        fcut = 0.5 * (jnp.cos(dist * jnp.pi / self.cutoff) + 1.0) * (dist < self.cutoff)
        wk = self.w["filter_w"][3 * F * k:3 * F * (k + 1)]
        bk = self.w["filter_b"][3 * F * k:3 * F * (k + 1)]
        W = (phi @ wk.T + bk) * fcut[:, None]
        m = W * xj
        dq, dmuR, dmumu = m[:, :F], m[:, F:2 * F], m[:, 2 * F:]
        direction = rij / dist[:, None]
        mu_j = hj[:, F:].reshape(-1, 3, F)
        dmu = dmuR[:, None, :] * direction[:, :, None] + dmumu[:, None, :] * mu_j
        return jnp.concatenate([dq, dmu.reshape(-1, 3 * F)], -1)

    def node(self, k, h, a):
        jnp = self.jnp
        F = self.F
        q = h[:, :F] + a[:, :F]
        mu = (h[:, F:] + a[:, F:]).reshape(-1, 3, F)
        mix = mu @ self.w[f"mix{k}_wmu"].T
        mu_V, mu_W = mix[..., :F], mix[..., F:]
        mu_Vn = jnp.sqrt(jnp.sum(mu_V ** 2, axis=1) + self.eps)
        y = self.linear(jnp.concatenate([q, mu_Vn], -1), f"mix{k}_w1", f"mix{k}_b1", act=True)
        y = self.linear(y, f"mix{k}_w2", f"mix{k}_b2")
        dq, dmu, dqmu = y[:, :F], y[:, F:2 * F], y[:, 2 * F:]
        q = q + dq + dqmu * jnp.sum(mu_V * mu_W, axis=1)
        mu = mu + dmu[:, None, :] * mu_W
        return jnp.concatenate([q, mu.reshape(-1, 3 * F)], -1)

    def readout(self, h):
        y = self.linear(h[:, :self.F], "out_w1", "out_b1", act=True)
        y = self.linear(y, "out_w2", "out_b2")
        return (y[:, 0] + self.mean) * HARTREE_KJ_PER_MOL

    def stages(self):
        """(name, function, number of differentiable operands, operands'
        shapes per row and types) of every stage."""
        F = self.F
        out = [("embed", self.embed, [("z", (), "i32")], False)]
        for k in range(self.L):
            out.append((f"ctx{k}", lambda h, k=k: self.ctx(k, h), [("h", (4 * F,), "f32")], True))
            out.append((f"edge{k}", lambda r, d, xj, hj, k=k: self.edge(k, r, d, xj, hj),
                        [("r", (), "f32"), ("d", (3,), "f32"), ("x", (3 * F,), "f32"),
                         ("h", (4 * F,), "f32")], True))
            out.append((f"node{k}", lambda h, a, k=k: self.node(k, h, a),
                        [("h", (4 * F,), "f32"), ("a", (4 * F,), "f32")], True))
        out.append(("readout", self.readout, [("h", (4 * F,), "f32")], True))
        return out


def vjp_of(f):
    import jax

    def b(*args):
        primals, cot = args[:-1], args[-1]
        _, pullback = jax.vjp(f, *primals)
        return pullback(cot)
    return b


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def relative(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = max(np.max(np.abs(b)), 1e-30)
    return float(np.max(np.abs(a - b)) / scale)


def check_stages(w, S):
    import jax.numpy as jnp
    worst = 0.0
    for name in sorted({k.split("_")[1] for k in w if k.startswith("ref_")}):
        count = len([k for k in w if k.startswith(f"ref_{name}_in")])
        ins = [jnp.asarray(w[f"ref_{name}_in{n}"]) for n in range(count)]
        outs = [w[f"ref_{name}_out0"]]
        cot = [w[f"ref_{name}_cot0"]]
        refs = [w[f"ref_{name}_vjp{n}"] for n in range(count)]
        if name.startswith("ctx"):
            f = lambda h, k=int(name[3:]): S.ctx(k, h)
        elif name.startswith("edge"):
            f = lambda r, d, x, h, k=int(name[4:]): S.edge(k, r, d, x, h)
        elif name.startswith("node"):
            f = lambda h, a, k=int(name[4:]): S.node(k, h, a)
        else:
            f = S.readout
        y = f(*ins)
        e_out = relative(y, outs[0])
        grads = vjp_of(f)(*ins, jnp.asarray(cot[0]))
        e_vjp = max(relative(g, r) for g, r in zip(grads, refs))
        worst = max(worst, e_out, e_vjp)
        print(f"stage {name:8s} forward {e_out:.2e}  vjp {e_vjp:.2e}  (relative to the largest value)")
    return worst


def pairs_of(positions, cell, cutoff):
    """Every directed pair (i, j, d_ij = x_i - x_j + images) within the
    cutoff, every image, in nm."""
    n = len(positions)
    inv = np.linalg.inv(cell)
    reach = [int(np.ceil(cutoff * np.linalg.norm(inv[:, a]))) for a in range(3)]
    I, J, D = [], [], []
    for a in range(-reach[0], reach[0] + 1):
        for b in range(-reach[1], reach[1] + 1):
            for c in range(-reach[2], reach[2] + 1):
                shift = np.array([a, b, c]) @ cell
                d = positions[:, None, :] - positions[None, :, :] - shift
                r = np.linalg.norm(d, axis=-1)
                mask = (r < cutoff) & (r > 0)
                i, j = np.nonzero(mask)
                I.append(i); J.append(j); D.append(d[i, j])
    return np.concatenate(I), np.concatenate(J), np.concatenate(D)


def compose(S, z, positions, cell):
    """The energy, forces, and virial from the stages: the forward over the
    layers, then MDIR's reverse rules (docs/ml4-ir.md, Section 4.1) with each
    stage's product. Fields and sums in float64, the bodies in float32."""
    import jax.numpy as jnp
    f32 = lambda a: jnp.asarray(np.asarray(a, np.float32))
    n = len(z)
    i, j, d = pairs_of(positions, cell, S.cutoff / ANGSTROM_PER_NM)
    r = np.linalg.norm(d, axis=1)
    h = [np.asarray(S.embed(jnp.asarray(z)), np.float64)]
    xs, as_ = [], []
    for k in range(S.L):
        x = np.asarray(S.ctx(k, f32(h[-1])), np.float64)
        m = np.asarray(S.edge(k, f32(r), f32(d), f32(x[j]), f32(h[-1][j])), np.float64)
        a = np.zeros_like(h[-1]); np.add.at(a, i, m)
        xs.append(x); as_.append(a)
        h.append(np.asarray(S.node(k, f32(h[-1]), f32(a)), np.float64))
    energy = float(np.sum(np.asarray(S.readout(f32(h[-1])), np.float64)))
    hbar = np.asarray(vjp_of(S.readout)(f32(h[-1]), f32(np.ones(n)))[0], np.float64)
    forces = np.zeros((n, 3))
    virial = np.zeros((3, 3))
    for k in reversed(range(S.L)):
        node = lambda hh, aa, k=k: S.node(k, hh, aa)
        hb, ab = vjp_of(node)(f32(h[k]), f32(as_[k]), f32(hbar))
        hbar = np.asarray(hb, np.float64)
        abar = np.asarray(ab, np.float64)
        edge = lambda rr, dd, xx, hh, k=k: S.edge(k, rr, dd, xx, hh)
        rb, db, xb, hjb = [np.asarray(t, np.float64) for t in vjp_of(edge)(
            f32(r), f32(d), f32(xs[k][j]), f32(h[k][j]), f32(abar[i]))]
        g = rb[:, None] * d / r[:, None] + db
        np.add.at(forces, i, -g)
        np.add.at(forces, j, g)
        virial -= np.einsum("pa,pb->ab", d, g)
        xbar = np.zeros_like(xs[k]); np.add.at(xbar, j, xb)
        np.add.at(hbar, j, hjb)
        ctx = lambda hh, k=k: S.ctx(k, hh)
        hbar += np.asarray(vjp_of(ctx)(f32(h[k]), f32(xbar))[0], np.float64)
    return energy, forces, virial, len(i)


def cmd_check(a):
    w = dict(np.load(a.weights, allow_pickle=True))
    S = Stages(w)
    worst = check_stages(w, S) if any(k.startswith("ref_") for k in w) else 0.0
    ok = worst < a.stage_tolerance
    for path in a.references:
        ref = np.load(path)
        positions, cell = ref["positions"], ref["cell"]
        z = np.array([8, 1, 1] * (len(positions) // 3), np.int32)
        e, f, v, npairs = compose(S, z, positions, cell)
        de = abs(e - float(ref["energy"])) / abs(float(ref["energy"]))
        df = relative(f, ref["forces"])
        dv = relative(v, ref["virial"])
        print(f"{os.path.basename(os.path.dirname(os.path.abspath(path)))}: {len(z)} atoms, "
              f"{npairs} directed pairs: E {e:.6f} kJ/mol (reference {float(ref['energy']):.6f}), "
              f"relative differences: energy {de:.2e}, forces {df:.2e} of "
              f"{np.max(np.abs(ref['forces'])):.1f}, virial {dv:.2e} of "
              f"{np.max(np.abs(ref['virial'])):.1f}")
        ok = ok and de < 1e-6 and df < 3e-5 and dv < 1e-5
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def mlir_type(shape, ty):
    return ty if not shape else f"vector<{shape[0]}x{ty}>"


def export_stage(fn, specs, rows, symbol, stablehlo_opt, workdir):
    """jax.export (through jit's lowering) of `fn` with `rows` rows, to
    StableHLO, then linalg on tensors; returns the functions' text with
    the entry renamed to `symbol`."""
    import jax
    import jax.numpy as jnp
    args = [jax.ShapeDtypeStruct((rows,) + shape,
                                 jnp.int32 if ty == "i32" else jnp.float32)
            for shape, ty in specs]
    text = jax.jit(fn).lower(*args).as_text()
    src = os.path.join(workdir, symbol + ".stablehlo.mlir")
    with open(src, "w") as f:
        f.write(text)
    out = subprocess.run([stablehlo_opt, src, "--stablehlo-legalize-to-linalg",
                          "--canonicalize", "--mlir-print-local-scope"],
                         check=True, capture_output=True, text=True).stdout
    if "stablehlo." in out:
        bad = sorted(set(re.findall(r"stablehlo\.[a-z_]+", out)))
        raise SystemExit(f"{symbol}: not legalized to linalg: {', '.join(bad)}")
    # The module's body, its functions renamed apart.
    body = out[out.index("{", out.index("module")) + 1:out.rstrip().rindex("}")]
    names = re.findall(r"func\.func (?:public |private )?@([\w.$-]+)", body)
    for name in names:
        new = symbol if name == "main" else f"{symbol}.{name}"
        body = re.sub(rf"@{re.escape(name)}\b", "@" + new, body)
    body = body.replace("func.func public @", "func.func @")
    body = re.sub(r" attributes \{[^{}]*\}", "", body, count=0)
    body = re.sub(r"\{jax\.[^{}]*\}", "", body)
    head = out[:out.index("module")]
    return head, body


def cmd_export(a):
    w = dict(np.load(a.weights, allow_pickle=True))
    S = Stages(w)
    os.makedirs(a.workdir, exist_ok=True)
    F = S.F
    heads, bodies, decls = [], [], []
    for name, fn, operands, differentiable in S.stages():
        specs = [(shape, ty) for _, shape, ty in operands]
        h, b = export_stage(fn, specs, a.rows, f"{name}.f", a.stablehlo_opt, a.workdir)
        heads.append(h); bodies.append(b)
        out_shape = () if name == "readout" else ((3 * F,) if name.startswith("ctx") else (4 * F,))
        sig_in = ", ".join(mlir_type(s, "f64" if t == "f32" else "i32") for s, t in specs)
        sig_out = mlir_type(out_shape, "f64")
        if differentiable:
            vspecs = specs + [(out_shape, "f32")]
            h, b = export_stage(vjp_of(fn), vspecs, a.rows, f"{name}.b", a.stablehlo_opt,
                                a.workdir)
            heads.append(h); bodies.append(b)
            decls.append(f"md.stage @{name} forward(@{name}.f) vjp(@{name}.b) "
                         f": ({sig_in}) -> {sig_out}")
        else:
            decls.append(f"md.stage @{name} forward(@{name}.f) "
                         f": ({sig_in}) -> {sig_out}")
    maps = "\n".join(sorted(set(l for h in heads for l in h.splitlines() if l.startswith("#"))))
    with open(a.output, "w") as f:
        f.write(f"// PaiNN in stages, written by scripts/mlip/painn-stages.py export\n")
        f.write(f"// F = {F}, L = {S.L}, cutoff {S.cutoff} A, {a.rows} rows a tile\n")
        f.write(maps + "\n")
        f.write("\n".join(decls) + "\n")
        for b in bodies:
            f.write(b + "\n")
    print(f"wrote {a.output}: {len(decls)} stages, {a.rows} rows a tile")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("weights")
    s.add_argument("checkpoint", nargs="?")
    s.add_argument("output")
    s.add_argument("--synthetic", type=int, nargs=2, metavar=("F", "L"))
    s = sub.add_parser("check")
    s.add_argument("weights")
    s.add_argument("references", nargs="*")
    s.add_argument("--stage-tolerance", type=float, default=1e-5)
    s = sub.add_parser("export")
    s.add_argument("weights")
    s.add_argument("output")
    s.add_argument("--rows", type=int, required=True)
    s.add_argument("--stablehlo-opt", default="stablehlo-opt")
    s.add_argument("--workdir", default=None)
    a = p.parse_args()
    if a.cmd == "weights":
        w = (synthetic_weights(*a.synthetic) if a.synthetic
             else weights_from_checkpoint(a.checkpoint))
        np.savez(a.output, **w)
        print(f"wrote {a.output}: F = {int(w['F'])}, L = {int(w['L'])}")
    elif a.cmd == "check":
        import jax
        jax.config.update("jax_platforms", "cpu")
        cmd_check(a)
    else:
        import jax
        jax.config.update("jax_platforms", "cpu")
        a.workdir = a.workdir or os.path.dirname(os.path.abspath(a.output))
        cmd_export(a)


if __name__ == "__main__":
    main()
