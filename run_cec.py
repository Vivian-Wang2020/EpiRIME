"""Run EpiRIME on the CEC2017 / CEC2022 benchmark functions.

Settings of the paper: budget 1000 * D function evaluations, population size 100, nucleus-pool ratio 0.3, 30 runs
per function; run r on function F uses the seed 2026 + 7 * F + 20 * r.

    python run_cec.py --suite 2017 --dim 30                       # all CEC2017 functions, 30 runs
    python run_cec.py --suite 2022 --dim 10 --funcs 1 6 --runs 5 --jobs 4 --out cec2022_d10.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from epirime import EpiRIME
from epirime.cec import get

FUNCS = {2017: [1] + list(range(3, 31)), 2022: list(range(1, 13))}
FE_PER_DIM = 1000
N_CURVE = 50


def seed_of(func, run, seed0=2026):
    return seed0 + 7 * func + 20 * run


def one_run(args):
    suite, func, dim, run, rho = args
    p = get(suite, func, dim)
    opt = EpiRIME(pop_size=100, max_fes=FE_PER_DIM * dim, seed=seed_of(func, run), rho=rho)
    res = opt.optimize(p.f, p.lb, p.ub, dim)
    grid = np.linspace(0, FE_PER_DIM * dim, N_CURVE + 1)[1:]
    idx = np.searchsorted(res.curve_fes, grid, side="right") - 1
    curve = np.where(idx >= 0, res.curve_fit[np.clip(idx, 0, None)], np.nan)
    return func, run, float(res.best_fit), curve.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=int, choices=(2017, 2022), required=True)
    ap.add_argument("--dim", type=int, required=True, help="30 or 50 for CEC2017, 10 or 20 for CEC2022")
    ap.add_argument("--funcs", type=int, nargs="+", help="function numbers (default: whole suite)")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--jobs", type=int, default=1, help="number of worker processes")
    ap.add_argument("--out", default=None, help="JSON file for the results")
    a = ap.parse_args()
    funcs = a.funcs or FUNCS[a.suite]
    jobs = [(a.suite, f, a.dim, r, a.rho) for f in funcs for r in range(a.runs)]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        out = list(ex.map(one_run, jobs))
    fitness, curves = {}, {}
    for f, r, fit, cv in out:
        fitness.setdefault(f"F{f}", [None] * a.runs)[r] = fit
        curves.setdefault(f"F{f}", [None] * a.runs)[r] = cv
    for f in funcs:
        v = np.array(fitness[f"F{f}"])
        print(f"CEC{a.suite} F{f:<2d} D{a.dim}: mean {v.mean():.6e}  std {v.std(ddof=1) if len(v) > 1 else 0:.3e}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(suite=a.suite, dim=a.dim, budget=FE_PER_DIM * a.dim, runs=a.runs, rho=a.rho,
                           fitness=fitness, curves=curves), fh)


if __name__ == "__main__":
    main()
