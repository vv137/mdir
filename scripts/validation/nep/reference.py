"""An independent evaluator of NEP4 models in PyTorch, for docs/nep.md
(D356, #583).

Written from the formulas of the papers (Fan et al., Phys. Rev. B 104,
104309, 2021; Fan, J. Phys.: Condens. Matter 34, 125902, 2022; Fan et al.,
J. Chem. Phys. 157, 114801, 2022; Song et al., Nat. Commun. 15, 10208, 2024)
and the layout of GPUMD's nep.txt, not from another program's code. It
evaluates the angular descriptors in two independent ways:

  moments   q_nl = sum_m |sum_j g_n(r_ij) Y_lm(u_ij)|^2, with orthonormal
            Y_lm built here from d^m P_l/dz^m and (x + i y)^m;
  triplets  q_nl = (2l+1)/(4 pi) sum_j sum_k g_n(r_ij) g_n(r_ik) P_l(u_ij . u_ik),
            j = k included (the addition theorem; docs/nep.md, Section 2).

The 4-body invariant is KAPPA tr(T^3), T = sum_j g_n (u u^T - I/3), and the
5-body invariant C5 |sum_j g_n u_j|^4 (docs/nep.md, Section 2.3). Forces and
the virial are the derivatives of the energy (autograd), the virial under a
homogeneous strain of the positions and the cell.

Precisions: 'double' (all f64), and 'mixed' as docs/nep.md, Section 6 plans
it: positions, displacements before rounding, and the sum of the atomic
energies in f64; the displacement of each pair, the descriptors, and the
network in f32.

Usage:
  reference.py MODEL REF.npz [--triplets]
REF.npz is written by calorine.py (positions, cells, symbols, and calorine's
energies, forces, virials, and descriptors). Plain nep4 models with one
radial and one angular cutoff only (no ZBL, no has_q_112 and later terms);
an orthorhombic cell whose edges are at least twice the radial cutoff.
"""
import math
import sys

import numpy as np
import torch

torch.set_num_threads(8)

# The normalizations of the 4-body and 5-body invariants as GPUMD defines
# them (Fan et al., J. Chem. Phys. 157, 114801, 2022): its C4B[0] and C5B[0],
# as numbers. tr(T^3) has the coefficient 1/36 of s_20^3, hence KAPPA.
C4B0 = -0.007499480826664
C5 = 0.026596810706114
KAPPA = 36.0 * C4B0


def read_nep(path):
    lines = [l.split() for l in open(path) if l.strip()]
    head = lines[0]
    if head[0] != "nep4":
        raise SystemExit(f"{head[0]}: this reference reads plain nep4 models only")
    ntypes = int(head[1])
    elems = head[2:2 + ntypes]
    cut = lines[1]
    if cut[0] != "cutoff" or len(cut) != 5:
        raise SystemExit("one radial and one angular cutoff only")
    nmax, bs, lm, ann = lines[2], lines[3], lines[4], lines[5]
    nr, na, kr, ka = int(nmax[1]), int(nmax[2]), int(bs[1]), int(bs[2])
    lmax = int(lm[1])
    flags = [int(x) for x in lm[2:]] + [0, 0]
    if any(flags[2:]):
        raise SystemExit("has_q_112, 123, 233, and 134 are not in this reference")
    has222, has1111 = flags[0], flags[1]
    nneu = int(ann[1])
    numL = lmax + bool(has222) + bool(has1111)
    dim = (nr + 1) + (na + 1) * numL
    vals = np.array([float(l[0]) for l in lines[6:]])
    T2 = ntypes * ntypes
    nann = (dim + 2) * nneu * ntypes + 1
    ncr, nca = T2 * (nr + 1) * (kr + 1), T2 * (na + 1) * (ka + 1)
    if len(vals) != nann + ncr + nca + dim:
        raise SystemExit(f"{len(vals)} parameters, expected {nann + ncr + nca + dim}")
    p = 0
    w0, b0, w1 = [], [], []
    for _ in range(ntypes):  # per species: w0 (neurons x dim), b0, w1
        w0.append(vals[p:p + nneu * dim].reshape(nneu, dim)); p += nneu * dim
        b0.append(vals[p:p + nneu]); p += nneu
        w1.append(vals[p:p + nneu]); p += nneu
    b1 = vals[p]; p += 1  # shared by the species
    # The coefficients are stored [basis][type pair], basis = n (K+1) + k and
    # type pair = t_center * ntypes + t_neighbor.
    cr = vals[p:p + ncr].reshape(-1, T2).T.reshape(ntypes, ntypes, nr + 1, kr + 1); p += ncr
    ca = vals[p:p + nca].reshape(-1, T2).T.reshape(ntypes, ntypes, na + 1, ka + 1); p += nca
    scaler = vals[p:p + dim]
    return dict(elems=elems, rc_r=float(cut[1]), rc_a=float(cut[2]), nr=nr, na=na, kr=kr, ka=ka,
                lmax=lmax, has222=has222, has1111=has1111, w0=np.stack(w0), b0=np.stack(b0),
                w1=np.stack(w1), b1=b1, cr=cr, ca=ca, scaler=scaler)


def neighbor_pairs(pos, L, rc):
    """Directed pairs (i, j) within rc and the image shift of each, minimum image."""
    d = pos[None, :, :] - pos[:, None, :]
    shift = -L * np.round(d / L)
    r = np.linalg.norm(d + shift, axis=-1)
    i, j = np.nonzero((r < rc) & (r > 0))
    return torch.as_tensor(i), torch.as_tensor(j), torch.as_tensor(shift[i, j])


def basis(r, rc, K):
    """f_k(r) = (T_k(x) + 1) f_c(r) / 2, x = 2 (r/rc - 1)^2 - 1, f_c = (1 + cos(pi r/rc))/2."""
    fc = torch.where(r < rc, 0.5 * torch.cos(math.pi * r / rc) + 0.5, torch.zeros_like(r))
    x = 2.0 * (r / rc - 1.0) ** 2 - 1.0
    T = [torch.ones_like(r), x]
    for _ in range(2, K + 1):
        T.append(2.0 * x * T[-1] - T[-2])
    return torch.stack([(T[k] + 1.0) * 0.5 * fc for k in range(K + 1)], -1)


def q_moments(u, g, i, n, l):
    """sum_m |A_lm|^2, A_lm = sum_j g Y_lm(u_j); m < 0 doubles m > 0 (|Y_l,-m| = |Y_lm|)."""
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    q = 0.0
    for m in range(l + 1):
        N = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m))
        P = np.polynomial.legendre.leg2poly(np.eye(l + 1)[l])
        Q = np.polynomial.polynomial.polyder(P, m) if m else P  # d^m P_l / dz^m
        Qz = sum(float(a) * z ** k for k, a in enumerate(Q))
        re, im = torch.ones_like(x), torch.zeros_like(x)
        for _ in range(m):  # (x + i y)^m
            re, im = re * x - im * y, re * y + im * x
        Are = torch.zeros(n, g.shape[1], dtype=g.dtype).index_add_(0, i, (N * Qz * re)[:, None] * g)
        Aim = torch.zeros(n, g.shape[1], dtype=g.dtype).index_add_(0, i, (N * Qz * im)[:, None] * g)
        q = q + (1.0 if m == 0 else 2.0) * (Are ** 2 + Aim ** 2)
    return q


def q_triplets(u, g, i, n, l):
    """(2l+1)/(4 pi) sum_{j,k} g_j g_k P_l(u_j . u_k), j = k included."""
    P = np.polynomial.legendre.leg2poly(np.eye(l + 1)[l])
    rows = []
    for a in range(n):
        sel = (i == a).nonzero().squeeze(1)
        c = u[sel] @ u[sel].T
        Pc = sum(float(b) * c ** k for k, b in enumerate(P))
        rows.append((2 * l + 1) / (4 * math.pi) * torch.einsum("jn,jk,kn->n", g[sel], Pc, g[sel]))
    return torch.stack(rows)


def energy(m, pos, L, types, dtype, strain=None, angular="moments"):
    """The energy (f64 sum of f`dtype` atomic energies) at positions `pos` (f64),
    optionally under the homogeneous strain 1 + strain of positions and cell."""
    n = len(types)
    t = torch.as_tensor(types)
    M = torch.eye(3, dtype=torch.float64) + (strain if strain is not None else 0.0)

    def displacements(rc):
        i, j, s = neighbor_pairs(pos.detach().numpy(), L, rc)
        d = (pos[j] - pos[i] + s) @ M.T  # f64
        d = d.to(dtype)                  # f32 from here in mixed
        return i, j, d, d.norm(dim=1)

    i, j, d, r = displacements(m["rc_r"])
    c = torch.as_tensor(m["cr"], dtype=dtype)[t[i], t[j]]
    g = torch.einsum("pnk,pk->pn", c, basis(r, m["rc_r"], m["kr"]))
    qs = [torch.zeros(n, m["nr"] + 1, dtype=dtype).index_add_(0, i, g)]

    i, j, d, r = displacements(m["rc_a"])
    u = d / r[:, None]
    c = torch.as_tensor(m["ca"], dtype=dtype)[t[i], t[j]]
    g = torch.einsum("pnk,pk->pn", c, basis(r, m["rc_a"], m["ka"]))
    q_l = q_moments if angular == "moments" else q_triplets
    qs += [q_l(u, g, i, n, l) for l in range(1, m["lmax"] + 1)]
    if m["has222"]:
        uu = u[:, :, None] * u[:, None, :] - torch.eye(3, dtype=dtype) / 3.0
        T = torch.zeros(n, m["na"] + 1, 3, 3, dtype=dtype).index_add_(0, i, g[:, :, None, None] * uu[:, None])
        qs.append(KAPPA * torch.einsum("anij,anjk,anki->an", T, T, T))
    if m["has1111"]:
        v = torch.zeros(n, m["na"] + 1, 3, dtype=dtype).index_add_(0, i, g[:, :, None] * u[:, None, :])
        qs.append(C5 * (v * v).sum(-1) ** 2)
    q = torch.cat(qs, dim=1) * torch.as_tensor(m["scaler"], dtype=dtype)
    w0 = torch.as_tensor(m["w0"], dtype=dtype)[t]
    b0 = torch.as_tensor(m["b0"], dtype=dtype)[t]
    w1 = torch.as_tensor(m["w1"], dtype=dtype)[t]
    E_i = (w1 * torch.tanh(torch.einsum("and,ad->an", w0, q) - b0)).sum(1) - m["b1"]
    return E_i.to(torch.float64).sum(), q.detach().to(torch.float64)


def evaluate(m, pos, cell, types, dtype, angular="moments"):
    L = np.diag(cell)
    if not np.allclose(cell, np.diag(L)) or (L < 2 * m["rc_r"]).any():
        raise SystemExit("an orthorhombic cell with edges of at least twice the cutoff only")
    x = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    eps = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    E, q = energy(m, x, L, types, dtype, eps, angular)
    gx, ge = torch.autograd.grad(E, (x, eps))
    return E.item(), -gx.numpy(), -ge.numpy(), q.numpy()  # energy, forces, virial, descriptors


def main():
    m = read_nep(sys.argv[1])
    ref = np.load(sys.argv[2])
    rows = []
    for f in range(int(ref["nframes"])):
        pos, cell = ref[f"pos{f}"], ref[f"cell{f}"]
        types = [m["elems"].index(s) for s in ref[f"sym{f}"]]
        n = len(types)
        E64, F64, W64, q64 = evaluate(m, pos, cell, types, torch.float64)
        E32, F32, W32, q32 = evaluate(m, pos, cell, types, torch.float32)
        Ec, Fc, Wc, qc = (ref[f"{k}{f}"] for k in ("energy", "forces", "virial", "desc"))
        r = {
            "dE_double/atom": abs(E64 - Ec) / n, "dF_double": np.abs(F64 - Fc).max(),
            "dW_double/atom": np.abs(W64 - Wc).max() / n, "dq_double": np.abs(q64 - qc).max(),
            "dE_mixed/atom": abs(E32 - E64) / n, "dF_mixed": np.abs(F32 - F64).max(),
            "rms_dF_mixed": np.sqrt(((F32 - F64) ** 2).mean()), "rms_F": np.sqrt((F64 ** 2).mean()),
            "dW_mixed/atom": np.abs(W32 - W64).max() / n, "dq_mixed": np.abs(q32 - q64).max(),
        }
        if "--triplets" in sys.argv:
            Et, Ft, Wt, qt = evaluate(m, pos, cell, types, torch.float64, angular="triplets")
            r.update({"dq_triplets": np.abs(qt - q64).max(), "dE_triplets/atom": abs(Et - E64) / n,
                      "dF_triplets": np.abs(Ft - F64).max()})
        rows.append(r)
        print(f"frame {f} ({n} atoms): " + " ".join(f"{k}={v:.2g}" for k, v in r.items()), flush=True)
    print("max over frames: " + " ".join(f"{k}={max(r[k] for r in rows):.2g}" for k in rows[0]))


if __name__ == "__main__":
    main()
