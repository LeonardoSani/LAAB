"""Decompose M1^2 = ||r_s - r_s'||^2 into norm-difference vs direction terms,
across depth, for the Mixed representation. Tests whether the large L2 gap is
driven by ||r|| norm-DIFFERENCE or by the cosine deviation amplified by ||r||.

M1^2 = (||r_s|| - ||r_s'||)^2  +  2||r_s|| ||r_s'|| (1 - cos(r_s, r_s'))
       [ norm-diff term ]         [ direction term = 2||r||||r'|| * M2 ]
"""
import itertools

import torch

from src.relative.cosine_map import relative_cosine
from src.store.anchors import AnchorStore

SEEDS = list(range(1, 11))
DEPTHS = [0, 8, 64, 512, float("inf")]

store = AnchorStore()


def mixed(s, t):
    # Mixed M_s^t(x) = cos( E_s(x), phi_anchors(s,t) ); E_s(x) = phi_data(s,0)
    return relative_cosine(store.phi_data(s, 0), store.phi_anchors(s, t))


print(f"{'t':>5} {'mean|r|':>8} {'M1':>7} {'M2':>9} {'normdiff%':>9} {'direction%':>10}")
for t in DEPTHS:
    reps = {s: mixed(s, t) for s in SEEDS}
    nd_tot = dir_tot = m1sq_tot = m2_tot = rnorm_tot = 0.0
    n = npairs = 0
    for s, s2 in itertools.combinations(SEEDS, 2):
        r, rp = reps[s], reps[s2]
        nr, nrp = r.norm(dim=1), rp.norm(dim=1)
        cos = (r * rp).sum(1) / (nr * nrp).clamp_min(1e-12)
        nd_tot += (nr - nrp).pow(2).sum().item()
        dir_tot += (2 * nr * nrp * (1 - cos)).sum().item()
        m1sq_tot += (r - rp).pow(2).sum(1).sum().item()
        m2_tot += (1 - cos).sum().item()
        rnorm_tot += nr.sum().item()
        n += r.shape[0]
    tlab = "inf" if t == float("inf") else int(t)
    print(f"{str(tlab):>5} {rnorm_tot/n:8.3f} {(m1sq_tot/n)**0.5:7.3f} "
          f"{m2_tot/n:9.5f} {100*nd_tot/m1sq_tot:9.1f} {100*dir_tot/m1sq_tot:10.1f}")
