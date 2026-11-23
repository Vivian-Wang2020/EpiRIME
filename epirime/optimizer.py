"""Epitaxial RIME algorithm (EpiRIME).

EpiRIME keeps the population structure, the dimension-wise soft-rime mask with probability E = sqrt(t / T), and
the positive greedy selection of RIME, and replaces the soft-rime search and the hard-rime puncture by two
strategies:

S1, droplet-accretion soft-rime search
    The covariance of the whole population is decomposed as cov(X) = V diag(lam) V^T. Every individual i draws a
    nucleus x_nuc(i) uniformly from the best round(rho * N) individuals and a droplet xi_i ~ N(0, I), and builds
        c_i = x_nuc(i) + R_i * sqrt(2) * V (sqrt(lam) * xi_i).
    The freezing intensity R_i ~ U(0, min(2 M_R, 1)) is drawn per individual. The memory M_R starts at 0.5 and is
    updated after each generation from the improved individuals k with weights w_k = f(x_k) - f(u_k):
        M_R <- (M_R + sum_k w_k R_k^2 / sum_k w_k R_k) / 2.
    Coordinate j of the trial individual is taken from c_i with probability E and from x_i otherwise.

S2, lattice-oriented puncture
    B holds the eigenvectors of the covariance of the best floor(N / 2) individuals. In this frame,
    z_i = B^T u_i and z_best = B^T x_best; every lattice coordinate is copied from the best individual with
    probability p_i = (2 / sqrt(N)) * rank_i / (N - 1), where rank_i is the fitness rank of the parent (0 = best),
    and the trial individual is mapped back as u_i = B z_i.

The run stops after max_fes function evaluations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    best_fit: float
    best_x: np.ndarray
    curve_fes: np.ndarray     # number of evaluations at each record
    curve_fit: np.ndarray     # best fitness at each record
    fes: int


class EpiRIME:
    """Minimize f over the box [lb, ub].

    Parameters
    ----------
    pop_size : population size N
    max_fes  : budget of function evaluations
    seed     : seed of numpy's default random generator
    rho      : nucleus-pool ratio; the nucleus pool holds the best round(rho * N) individuals
    """

    name = "EpiRIME"

    def __init__(self, pop_size=100, max_fes=300_000, seed=None, rho=0.3):
        self.N = pop_size
        self.max_fes = max_fes
        self.rng = np.random.default_rng(seed)
        self.rho = rho

    # ------------------------------------------------------------------ bookkeeping
    def _reset(self):
        self.fes = 0
        self._cf, self._cv = [], []
        self.best_fit, self.best_x = np.inf, None

    def _eval(self, fobj, X):
        """Evaluate a batch of rows; rows beyond the remaining budget are not evaluated and receive inf."""
        X = np.atleast_2d(X)
        n = min(X.shape[0], self.max_fes - self.fes)
        fit = np.full(X.shape[0], np.inf)
        if n > 0:
            fit[:n] = fobj(X[:n])
            self.fes += n
            b = int(np.argmin(fit[:n]))
            if fit[b] < self.best_fit:
                self.best_fit, self.best_x = fit[b], X[b].copy()
            self._cf.append(self.fes)
            self._cv.append(self.best_fit)
        return fit

    @property
    def budget_left(self):
        return self.max_fes - self.fes

    # ------------------------------------------------------------------ search
    def optimize(self, fobj, lb, ub, dim) -> Result:
        """fobj maps an (n, dim) array to n fitness values; lb and ub are scalars or arrays of length dim."""
        lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, float)
        ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, float)
        self._reset()
        N, D, rng = self.N, dim, self.rng
        q = int(round(self.rho * N))                       # size of the nucleus pool
        T = max(1, (self.max_fes - N) // N)                 # number of generations allowed by the budget
        X = lb + rng.random((N, D)) * (ub - lb)
        fit = self._eval(fobj, X)
        M_R = 0.5
        rows = np.arange(N)
        t = 0

        while self.budget_left > 0:
            t += 1
            order = np.argsort(fit)
            rank = np.empty(N)
            rank[order] = rows
            nuc = order[rng.integers(0, q, N)]
            C = np.atleast_2d(np.cov(X[order[:N // 2]], rowvar=False))
            try:
                _, B = np.linalg.eigh(C)                     # lattice frame of the elite individuals
            except np.linalg.LinAlgError:
                B = np.eye(D)

            # S1: droplets shaped by the population covariance, attached to the nuclei
            R = min(2 * M_R, 1.0) * rng.random(N)
            try:
                lam, V = np.linalg.eigh(np.atleast_2d(np.cov(X, rowvar=False)))
            except np.linalg.LinAlgError:
                lam, V = np.ones(D), np.eye(D)
            xi = rng.standard_normal((N, D))
            droplet = np.sqrt(2.0) * (xi * np.sqrt(np.maximum(lam, 0.0))) @ V.T
            cand = X[nuc] + R[:, None] * droplet
            mask = rng.random((N, D)) < np.sqrt(min(t, T) / T)  # soft-rime mask of RIME
            Xn = X.copy()
            Xn[mask] = cand[mask]

            # S2: puncture in the lattice frame, rank-based probability
            p = (2.0 / np.sqrt(N)) * rank / (N - 1)
            m2 = rng.random((N, D)) < p[:, None]
            Z, Zb = Xn @ B, np.broadcast_to(X[order[0]], X.shape) @ B   # rows: z_i^T = u_i^T B
            Z[m2] = Zb[m2]
            Xn = Z @ B.T

            # keep the trial individuals inside the box
            Xn = np.where(Xn < lb, (X + lb) / 2, Xn)
            Xn = np.where(Xn > ub, (X + ub) / 2, Xn)

            nf = self._eval(fobj, Xn)
            better = nf < fit
            if better.any():
                w, Rs = fit[better] - nf[better], R[better]
                if w @ Rs > 0:
                    M_R = 0.5 * M_R + 0.5 * float((w @ Rs ** 2) / (w @ Rs))
            X[better], fit[better] = Xn[better], nf[better]

        return Result(self.best_fit, self.best_x, np.array(self._cf), np.array(self._cv), self.fes)
