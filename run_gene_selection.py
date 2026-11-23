"""Run EpiRIME for gene selection on the head and neck cancer datasets.

Settings of the paper: budget 20,000 fitness evaluations, population size 100, nucleus-pool ratio 0.3, 30 runs per
dataset; run r on dataset number k (in the order of DATASETS) uses the data split seed 7,000,000 + 1000 k + r and
the optimizer seed 8,000,000 + 1000 k + r. For every run, the script reports the training fitness, the number of
selected genes, the training balanced error, and the balanced error on the held-out test patients.

    python run_gene_selection.py --datasets NPC-GSE53819 --runs 3
    python run_gene_selection.py --runs 30 --jobs 8 --out gene_selection.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from epirime import EpiRIME
from epirime.gene_selection import DATASETS, GeneSelection, alg_seed


def one_run(args):
    name, run, fes, rho = args
    p = GeneSelection(name, run)
    opt = EpiRIME(pop_size=100, max_fes=fes, seed=alg_seed(p.ds_idx, run), rho=rho)
    res = opt.optimize(p.f, p.lb, p.ub, p.dim)
    m = p.mask(res.best_x)
    return dict(dataset=name, run=run, fitness=float(res.best_fit), n_genes=int(m.sum()),
                train_error=p.train_error(m), test_error=p.test_error(res.best_x),
                selected=[str(x) for x in p.probes[m]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--fes", type=int, default=20_000)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--jobs", type=int, default=1, help="number of worker processes")
    ap.add_argument("--out", default=None, help="JSON file for the results")
    a = ap.parse_args()
    jobs = [(d, r, a.fes, a.rho) for d in a.datasets for r in range(a.runs)]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        out = list(ex.map(one_run, jobs))
    for d in a.datasets:
        rows = [r for r in out if r["dataset"] == d]
        mean = lambda k: np.mean([r[k] for r in rows])
        print(f"{d:16s} fitness {mean('fitness'):.3e}  genes {mean('n_genes'):7.1f}  "
              f"train BE {mean('train_error'):.3f}  test BE {mean('test_error'):.3f}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(budget=a.fes, rho=a.rho, runs=out), fh)


if __name__ == "__main__":
    main()
