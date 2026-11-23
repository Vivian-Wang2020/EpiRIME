"""Wrapper gene selection on head and neck cancer expression data (tumor vs. non-malignant tissue).

Data: data/geo/<NAME>.npz, built by scripts/prepare_geo.py, with the 2000 probes of largest variance per dataset.
Encoding: x in [0, 1]^D with D = 2000; gene j is selected when x_j > 0.5.
Fitness (minimized): F(x) = 0.99 * BE(S) + 0.01 * |S| / D, where BE(S) is the balanced error of a k-nearest-neighbor
classifier with k = 3 on the training samples, computed leave-one-patient-out: the neighbors of a sample never
include the sample itself or other samples of the same patient. An empty subset receives F = 1.
Splitting: each run splits the patients into 70% training and 30% test data, stratified by class, so that all
samples of a patient fall on the same side. Standardization uses the training samples only, and the test samples
are used only after the search, by test_error().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "geo"
DATASETS = ["NPC-GSE12452", "NPC-GSE53819", "LSCC-GSE59102", "LSCC-GSE143224", "HNSCC-GSE107591",
            "OSCC-GSE30784", "OSCC-GSE25099", "OSCC-GSE37991", "OSCC-GSE13601"]
K_NN = 3
ALPHA = 0.99
SPLIT_SEED0 = 7_000_000
ALG_SEED0 = 8_000_000


def split_seed(ds_idx: int, run: int) -> int:
    """Seed of the data split of run `run` on dataset number `ds_idx` (shared by all algorithms)."""
    return SPLIT_SEED0 + 1000 * ds_idx + run


def alg_seed(ds_idx: int, run: int) -> int:
    """Seed of the optimizer in run `run` on dataset number `ds_idx`."""
    return ALG_SEED0 + 1000 * ds_idx + run


def grouped_stratified_split(y, groups, seed, test_frac=0.3):
    """Assign whole patients to the test set so that about test_frac of the patients of each class are held out
    (the class of a patient is the majority class of its samples)."""
    rng = np.random.default_rng(seed)
    y, groups = np.asarray(y), np.asarray(groups)
    ug = np.unique(groups)
    glab = np.array([int(round(y[groups == g].mean())) for g in ug])
    test_groups = []
    for c in (0, 1):
        gc = ug[glab == c]
        gc = gc[rng.permutation(len(gc))]
        test_groups.extend(gc[:int(round(test_frac * len(gc)))].tolist())
    te = np.isin(groups, test_groups)
    return np.where(~te)[0], np.where(te)[0]


def balanced_error(y_true, y_pred):
    return float(np.mean([np.mean(y_pred[y_true == c] != c) for c in np.unique(y_true)]))


class GeneSelection:
    """One run of one dataset: dim, lb, ub, f(X) -> (n,), plus train_error() and test_error() for reporting."""

    def __init__(self, name: str, run: int, data_dir=DATA):
        self.dataset, self.run = name, int(run)
        self.ds_idx = DATASETS.index(name)
        d = np.load(Path(data_dir) / f"{name}.npz", allow_pickle=False)
        X, y, g = d["X"].astype(float), d["y"].astype(int), d["groups"]
        self.probes, self.symbols = d["feat"], d["sym"]
        tr, te = grouped_stratified_split(y, g, split_seed(self.ds_idx, run))
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        self.Xtr, self.ytr, self.Xte, self.yte = Z[tr], y[tr], Z[te], y[te]
        gtr = np.asarray(g)[tr]
        self.same = gtr[:, None] == gtr[None, :]        # a sample and samples of the same patient are no neighbors
        self.dim = X.shape[1]
        self.lb, self.ub = np.zeros(self.dim), np.ones(self.dim)
        self._sq = self.Xtr ** 2

    @staticmethod
    def mask(x):
        return np.asarray(x) > 0.5

    def train_error(self, m) -> float:
        """Leave-one-patient-out balanced error of the KNN classifier on the training samples."""
        if not m.any():
            return 1.0
        A = self.Xtr[:, m]
        n2 = self._sq[:, m].sum(axis=1)
        D2 = n2[:, None] + n2[None, :] - 2.0 * (A @ A.T)
        D2[self.same] = np.inf
        idx = np.argsort(D2, axis=1, kind="stable")[:, :K_NN]
        pred = (self.ytr[idx].sum(axis=1) * 2 > K_NN).astype(int)
        return balanced_error(self.ytr, pred)

    def fitness_of_mask(self, m) -> float:
        if not m.any():
            return 1.0
        return ALPHA * self.train_error(m) + (1.0 - ALPHA) * m.sum() / self.dim

    def f(self, X):
        X = np.atleast_2d(X)
        return np.array([self.fitness_of_mask(self.mask(x)) for x in X])

    __call__ = f

    def test_error(self, x) -> float:
        """Balanced error on the held-out test samples of the KNN classifier that uses all training samples."""
        m = self.mask(x)
        if not m.any():
            return 1.0
        A, B = self.Xtr[:, m], self.Xte[:, m]
        D2 = (B ** 2).sum(1)[:, None] + (A ** 2).sum(1)[None, :] - 2.0 * (B @ A.T)
        idx = np.argsort(D2, axis=1, kind="stable")[:, :K_NN]
        pred = (self.ytr[idx].sum(axis=1) * 2 > K_NN).astype(int)
        return balanced_error(self.yte, pred)
