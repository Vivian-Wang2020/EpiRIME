"""CEC2017 and CEC2022 bound-constrained single-objective benchmark functions in numpy, vectorized over rows.

The functions follow the official C implementations of the two competitions and read the official shift,
rotation, and shuffle data in cec_data/. The behavior of the official code is reproduced, including the following
details, so that the values agree with the published C library:

  * Schaffer F7 is evaluated on the shifted and scaled vector before rotation (CEC2017 F6, CEC2022 F3), and inside
    the hybrid functions CEC2017 F14, F20 and CEC2022 F7 on the first n_k components of the permuted vector;
  * the Lunacek bi-Rastrigin function takes its signs from the first n entries of the shift vector, also inside
    the hybrid function CEC2017 F13;
  * CEC2017 F8 and CEC2022 F4 are shifted and rotated Rastrigin functions;
  * the Levy function uses sin(pi * w + 1) in the sum, with w = 1 + (z - 1) / 4 in CEC2017 and w = 1 + z / 4 in
    CEC2022;
  * the group sizes of the hybrid functions are ceil(p * D) in double precision.

CEC2017 F2 was removed from the suite by the organizers and is not used. Only the data for the dimensions used in
the experiments are included: D = 30 and 50 for CEC2017 and D = 10 and 20 for CEC2022.

Usage: f = get(2017, 5, 30); values = f(X) for X of shape (n, 30); f.lb, f.ub, f.optimum.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cec_data")
PI = math.pi
E = math.e
INF = 1.0e99
TWO_PI = 2.0 * PI
CEC22_FOPT = (300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700)
DIMS = {2017: (2, 10, 20, 30, 50, 100), 2022: (2, 10, 20)}


# ----------------------------------------------------------------------------- data (fscanf semantics)
def _tokens(path):
    with open(path) as fh:
        return tuple(fh.read().split())


def _lines(path):
    with open(path) as fh:
        return tuple(ln.split() for ln in fh.read().splitlines() if ln.strip())


def _vec(path, n, dtype=float):
    t = _tokens(path)
    if len(t) < n:
        raise ValueError(f"{path}: {len(t)} values, {n} needed")
    return np.array(t[:n], dtype=dtype)


def _rows(path, k, n):
    """k rows of n values: first n values of each of the first k lines (the C code's '%*[^\\n]%*c' skip)."""
    L = _lines(path)
    if len(L) < k:
        raise ValueError(f"{path}: {len(L)} lines, {k} needed")
    return np.array([ln[:n] for ln in L[:k]], dtype=float)


# ----------------------------------------------------------------------------- C-global emulation
class _Ctx:
    """Holds the C global y (n, D) between calls: Schaffer F7 and bi-Rastrigin read / write it."""
    __slots__ = ("y",)

    def __init__(self, n, D):
        self.y = np.zeros((n, D))


def _rot(y, M):
    """rotatefunc: xrot[i] = sum_j x[j] * M[i][j], accumulated in j order (as the C loop)."""
    acc = y[:, 0:1] * M[:, 0][None, :]
    for j in range(1, y.shape[1]):
        acc = acc + y[:, j:j + 1] * M[:, j][None, :]
    return acc


def _sr(x, Os, Mr, rate, s, r, ctx):
    """sr_func: returns z; writes ctx.y where the C code writes the global y."""
    m = x.shape[1]
    if s:
        if r:
            y = (x - Os[None, :m]) * rate
            ctx.y[:, :m] = y
            return _rot(y, Mr)
        return (x - Os[None, :m]) * rate
    if r:
        y = x * rate
        ctx.y[:, :m] = y
        return _rot(y, Mr)
    return x * rate


def _cumsum_cols(a):
    """sequential sum over columns (C loop order)."""
    s = a[:, 0].copy()
    for j in range(1, a.shape[1]):
        s = s + a[:, j]
    return s


# ----------------------------------------------------------------------------- basic functions
# signature: f(x, Os, Mr, s_flag, r_flag, ctx) -> (n,)
def _ellips(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 1.0, s, r, ctx)
    m = z.shape[1]
    f = np.zeros(z.shape[0])
    for i in range(m):
        c = math.pow(10.0, 6.0 * i / (m - 1)) if m > 1 else math.nan
        f = f + c * z[:, i] * z[:, i]
    return f


def _bent_cigar(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 1.0, s, r, ctx)
    f = z[:, 0] * z[:, 0]
    for i in range(1, z.shape[1]):
        f = f + 1.0e6 * z[:, i] * z[:, i]
    return f


def _discus(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 1.0, s, r, ctx)
    f = 1.0e6 * z[:, 0] * z[:, 0]
    for i in range(1, z.shape[1]):
        f = f + z[:, i] * z[:, i]
    return f


def _sum_diff_pow(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 1.0, s, r, ctx)
    f = np.zeros(z.shape[0])
    with np.errstate(over="ignore"):
        for i in range(z.shape[1]):
            f = f + np.power(np.abs(z[:, i]), float(i + 1))
    return f


def _zakharov(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 1.0, s, r, ctx)
    s1 = np.zeros(z.shape[0])
    s2 = np.zeros(z.shape[0])
    for i in range(z.shape[1]):
        s1 = s1 + z[:, i] * z[:, i]
        s2 = s2 + 0.5 * (i + 1) * z[:, i]
    with np.errstate(over="ignore"):
        return s1 + s2 * s2 + np.power(s2, 4.0)


def _levy(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 1.0, s, r, ctx)
    m = z.shape[1]
    w = 1.0 + (z - 1.0) / 4.0 if suite == 2017 else 1.0 + (z - 0.0) / 4.0
    t1 = np.sin(PI * w[:, 0]) ** 2
    t3 = (w[:, m - 1] - 1) ** 2 * (1 + np.sin(TWO_PI * w[:, m - 1]) ** 2)
    sm = np.zeros(z.shape[0])
    for i in range(m - 1):
        wi = w[:, i]
        sm = sm + (wi - 1) ** 2 * (1 + 10 * np.sin(PI * wi + 1) ** 2)
    return t1 + sm + t3


def _rosenbrock(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 2.048 / 100.0, s, r, ctx) + 1.0
    f = np.zeros(z.shape[0])
    for i in range(z.shape[1] - 1):
        t1 = z[:, i] * z[:, i] - z[:, i + 1]
        t2 = z[:, i] - 1.0
        f = f + 100.0 * t1 * t1 + t2 * t2
    return f


def _schaffer_f7(x, Os, Mr, s, r, ctx, suite=2017):
    _sr(x, Os, Mr, 1.0, s, r, ctx)
    m = x.shape[1]
    y = ctx.y  # official quirk: reads the global y, not z
    f = np.zeros(x.shape[0])
    for i in range(m - 1):
        zi = np.sqrt(y[:, i] * y[:, i] + y[:, i + 1] * y[:, i + 1])
        t = np.sin(50.0 * np.power(zi, 0.2))
        sq = np.sqrt(zi)
        f = f + (sq + sq * t * t)
    with np.errstate(divide="ignore", invalid="ignore"):
        return f * f / (m - 1) / (m - 1) if m > 1 else f * f / 0.0


def _ackley(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 1.0, s, r, ctx)
    m = z.shape[1]
    s1 = np.zeros(z.shape[0])
    s2 = np.zeros(z.shape[0])
    for i in range(m):
        s1 = s1 + z[:, i] * z[:, i]
        s2 = s2 + np.cos(TWO_PI * z[:, i])
    s1 = -0.2 * np.sqrt(s1 / m)
    s2 = s2 / m
    return E - 20.0 * np.exp(s1) - np.exp(s2) + 20.0


_WA = [math.pow(0.5, j) for j in range(21)]
_WB = [TWO_PI * math.pow(3.0, j) for j in range(21)]
_WS2 = 0.0
for _j in range(21):
    _WS2 += _WA[_j] * math.cos(_WB[_j] * 0.5)


def _weierstrass(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 0.5 / 100.0, s, r, ctx)
    m = z.shape[1]
    f = np.zeros(z.shape[0])
    for i in range(m):
        sm = np.zeros(z.shape[0])
        zi = z[:, i] + 0.5
        for j in range(21):
            sm = sm + _WA[j] * np.cos(_WB[j] * zi)
        f = f + sm
    return f - m * _WS2


def _griewank(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 600.0 / 100.0, s, r, ctx)
    sm = np.zeros(z.shape[0])
    p = np.ones(z.shape[0])
    for i in range(z.shape[1]):
        sm = sm + z[:, i] * z[:, i]
        p = p * np.cos(z[:, i] / math.sqrt(1.0 + i))
    return 1.0 + sm / 4000.0 - p


def _rastrigin(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 5.12 / 100.0, s, r, ctx)
    f = np.zeros(z.shape[0])
    for i in range(z.shape[1]):
        f = f + (z[:, i] * z[:, i] - 10.0 * np.cos(TWO_PI * z[:, i]) + 10.0)
    return f


_step_rastrigin = _rastrigin  # the official discretisation step is overwritten by sr_func (dead code)


def _schwefel(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 1000.0 / 100.0, s, r, ctx) + 4.209687462275036e+002
    m = z.shape[1]
    f = np.zeros(z.shape[0])
    for i in range(m):
        zi = z[:, i]
        hi, lo = zi > 500, zi < -500
        mid = ~(hi | lo)
        a = 500.0 - np.fmod(zi[hi], 500)
        t = (zi[hi] - 500.0) / 100
        term_hi = -(a * np.sin(np.sqrt(a)))
        fz = np.fmod(np.abs(zi[lo]), 500)
        t2 = (zi[lo] + 500.0) / 100
        term_lo = -((-500.0 + fz) * np.sin(np.sqrt(500.0 - fz)))
        # keep the C operation order: f -= a*sin(..); f += tmp*tmp/nx
        fn = f.copy()
        fn[hi] = (f[hi] + term_hi) + t * t / m
        fn[lo] = (f[lo] + term_lo) + t2 * t2 / m
        fn[mid] = f[mid] - zi[mid] * np.sin(np.sqrt(np.abs(zi[mid])))
        f = fn
    return f + 4.189828872724338e+002 * m


def _katsuura(x, Os, Mr, s, r, ctx, suite=2017):
    m = x.shape[1]
    tmp3 = math.pow(1.0 * m, 1.2)
    z = _sr(x, Os, Mr, 5.0 / 100.0, s, r, ctx)
    f = np.ones(z.shape[0])
    ex = 10.0 / tmp3
    for i in range(m):
        temp = np.zeros(z.shape[0])
        for j in range(1, 33):
            t1 = math.pow(2.0, j)
            t2 = t1 * z[:, i]
            temp = temp + np.abs(t2 - np.floor(t2 + 0.5)) / t1
        f = f * np.power(1.0 + (i + 1) * temp, ex)
    t1 = 10.0 / m / m
    return f * t1 - t1


def _bi_rastrigin(x, Os, Mr, s, r, ctx, suite=2017):
    m = x.shape[1]
    mu0, d = 2.5, 1.0
    sv = 1.0 - 1.0 / (2.0 * math.pow(m + 20.0, 0.5) - 8.2)
    mu1 = -math.pow((mu0 * mu0 - d) / sv, 0.5)
    y = (x - Os[None, :m]) if s else x.copy()
    y = y * (10.0 / 100.0)
    ctx.y[:, :m] = y
    tmpx = 2 * y
    neg = Os[:m] < 0.0
    tmpx[:, neg] = tmpx[:, neg] * -1.0
    z = tmpx
    tx = tmpx + mu0
    t1 = np.zeros(x.shape[0])
    t2 = np.zeros(x.shape[0])
    for i in range(m):
        a = tx[:, i] - mu0
        t1 = t1 + a * a
        b = tx[:, i] - mu1
        t2 = t2 + b * b
    t2 = t2 * sv
    t2 = t2 + d * m
    if r:
        yy = _rot(z, Mr)
        ctx.y[:, :m] = yy
        c = _cumsum_cols(np.cos(TWO_PI * yy))
    else:
        c = _cumsum_cols(np.cos(TWO_PI * z))
    return np.where(t1 < t2, t1, t2) + 10.0 * (m - c)


def _grie_rosen(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 5.0 / 100.0, s, r, ctx) + 1.0
    m = z.shape[1]
    f = np.zeros(z.shape[0])
    for i in range(m - 1):
        t1 = z[:, i] * z[:, i] - z[:, i + 1]
        t2 = z[:, i] - 1.0
        tp = 100.0 * t1 * t1 + t2 * t2
        f = f + (tp * tp / 4000.0 - np.cos(tp) + 1.0)
    t1 = z[:, m - 1] * z[:, m - 1] - z[:, 0]
    t2 = z[:, m - 1] - 1.0
    tp = 100.0 * t1 * t1 + t2 * t2
    return f + (tp * tp / 4000.0 - np.cos(tp) + 1.0)


def _escaffer6(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 1.0, s, r, ctx)
    m = z.shape[1]
    f = np.zeros(z.shape[0])

    def term(a, b):
        q = a * a + b * b
        t1 = np.sin(np.sqrt(q))
        t1 = t1 * t1
        t2 = 1.0 + 0.001 * q
        return 0.5 + (t1 - 0.5) / (t2 * t2)
    for i in range(m - 1):
        f = f + term(z[:, i], z[:, i + 1])
    return f + term(z[:, m - 1], z[:, 0])


def _happycat(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 5.0 / 100.0, s, r, ctx) - 1.0
    m = z.shape[1]
    r2 = _cumsum_cols(z * z)
    sz = _cumsum_cols(z)
    return np.power(np.abs(r2 - m), 0.25) + (0.5 * r2 + sz) / m + 0.5


def _hgbat(x, Os, Mr, s, r, ctx, suite=2017):
    z = _sr(x, Os, Mr, 5.0 / 100.0, s, r, ctx) - 1.0
    m = z.shape[1]
    r2 = _cumsum_cols(z * z)
    sz = _cumsum_cols(z)
    return np.sqrt(np.abs(r2 * r2 - sz * sz)) + (0.5 * r2 + sz) / m + 0.5


# ----------------------------------------------------------------------------- hybrid / composition
def _hybrid(x, Os, Mr, S, s, r, ctx, Gp, funcs, suite):
    nx = x.shape[1]
    k = len(Gp)
    Gn = [int(math.ceil(Gp[i] * nx)) for i in range(k - 1)]
    Gn.append(nx - sum(Gn))
    G = [0]
    for i in range(1, k):
        G.append(G[-1] + Gn[i - 1])
    z = _sr(x, Os, Mr, 1.0, s, r, ctx)
    ctx.y[:, :nx] = z[:, S - 1]
    f = np.zeros(x.shape[0])
    fits = []
    for i, fn in enumerate(funcs):
        sub = ctx.y[:, G[i]:G[i] + Gn[i]].copy()
        fits.append(fn(sub, Os, Mr, 0, 0, ctx, suite))
    for v in fits:
        f = f + v
    return f


def _cf_cal(x, Os, delta, bias, fit):
    nx = x.shape[1]
    k = len(fit)
    W = []
    wmax = np.zeros(x.shape[0])
    for i in range(k):
        fit[i] = fit[i] + bias[i]
        d = x - Os[i][None, :nx]
        w = _cumsum_cols(d * d)
        with np.errstate(divide="ignore", over="ignore", under="ignore"):
            wv = np.sqrt(1.0 / w) * np.exp(-w / 2.0 / nx / (delta[i] * delta[i]))
        wv = np.where(w != 0, wv, INF)
        wmax = np.where(wv > wmax, wv, wmax)
        W.append(wv)
    wsum = W[0].copy()
    for i in range(1, k):
        wsum = wsum + W[i]
    zero = wmax == 0
    if zero.any():
        for i in range(k):
            W[i] = np.where(zero, 1.0, W[i])
        wsum = np.where(zero, float(k), wsum)
    f = np.zeros(x.shape[0])
    for i in range(k):
        f = f + W[i] / wsum * fit[i]
    return f


# composition component: (function, scale (num, den) or None, r_flag override or None)
def _sc(num, den):
    return (float(num), float(den))


CF17 = {
    21: ([(_rosenbrock, None), (_ellips, _sc(10000, 1e10)), (_rastrigin, None)], [10, 20, 30], [0, 100, 200]),
    22: ([(_rastrigin, None), (_griewank, _sc(1000, 100)), (_schwefel, None)], [10, 20, 30], [0, 100, 200]),
    23: ([(_rosenbrock, None), (_ackley, _sc(1000, 100)), (_schwefel, None), (_rastrigin, None)],
         [10, 20, 30, 40], [0, 100, 200, 300]),
    24: ([(_ackley, _sc(1000, 100)), (_ellips, _sc(10000, 1e10)), (_griewank, _sc(1000, 100)), (_rastrigin, None)],
         [10, 20, 30, 40], [0, 100, 200, 300]),
    25: ([(_rastrigin, _sc(10000, 1e3)), (_happycat, _sc(1000, 1e3)), (_ackley, _sc(1000, 100)),
          (_discus, _sc(10000, 1e10)), (_rosenbrock, None)], [10, 20, 30, 40, 50], [0, 100, 200, 300, 400]),
    26: ([(_escaffer6, _sc(10000, 2e7)), (_schwefel, None), (_griewank, _sc(1000, 100)), (_rosenbrock, None),
          (_rastrigin, _sc(10000, 1e3))], [10, 20, 20, 30, 40], [0, 100, 200, 300, 400]),
    27: ([(_hgbat, _sc(10000, 1000)), (_rastrigin, _sc(10000, 1e3)), (_schwefel, _sc(10000, 4e3)),
          (_bent_cigar, _sc(10000, 1e30)), (_ellips, _sc(10000, 1e10)), (_escaffer6, _sc(10000, 2e7))],
         [10, 20, 30, 40, 50, 60], [0, 100, 200, 300, 400, 500]),
    28: ([(_ackley, _sc(1000, 100)), (_griewank, _sc(1000, 100)), (_discus, _sc(10000, 1e10)), (_rosenbrock, None),
          (_happycat, _sc(1000, 1e3)), (_escaffer6, _sc(10000, 2e7))],
         [10, 20, 30, 40, 50, 60], [0, 100, 200, 300, 400, 500]),
}
CF22 = {  # (function, scale, r_flag)
    9: ([(_rosenbrock, _sc(10000, 1e4), 1), (_ellips, _sc(10000, 1e10), 1), (_bent_cigar, _sc(10000, 1e30), 1),
         (_discus, _sc(10000, 1e10), 1), (_ellips, _sc(10000, 1e10), 0)], [10, 20, 30, 40, 50], [0, 200, 300, 100, 400]),
    10: ([(_schwefel, None, 0), (_rastrigin, None, 1), (_hgbat, None, 1)], [20, 10, 10], [0, 200, 100]),
    11: ([(_escaffer6, _sc(10000, 2e7), 1), (_schwefel, None, 1), (_griewank, _sc(1000, 100), 1), (_rosenbrock, None, 1),
          (_rastrigin, _sc(10000, 1e3), 1)], [20, 20, 30, 30, 20], [0, 200, 300, 400, 200]),
    12: ([(_hgbat, _sc(10000, 1000), 1), (_rastrigin, _sc(10000, 1e3), 1), (_schwefel, _sc(10000, 4e3), 1),
          (_bent_cigar, _sc(10000, 1e30), 1), (_ellips, _sc(10000, 1e10), 1), (_escaffer6, _sc(10000, 2e7), 1)],
         [10, 20, 30, 40, 50, 60], [0, 300, 500, 100, 400, 200]),
}

HF17 = {
    11: ([0.2, 0.4, 0.4], [_zakharov, _rosenbrock, _rastrigin]),
    12: ([0.3, 0.3, 0.4], [_ellips, _schwefel, _bent_cigar]),
    13: ([0.3, 0.3, 0.4], [_bent_cigar, _rosenbrock, _bi_rastrigin]),
    14: ([0.2, 0.2, 0.2, 0.4], [_ellips, _ackley, _schaffer_f7, _rastrigin]),
    15: ([0.2, 0.2, 0.3, 0.3], [_bent_cigar, _hgbat, _rastrigin, _rosenbrock]),
    16: ([0.2, 0.2, 0.3, 0.3], [_escaffer6, _hgbat, _rosenbrock, _schwefel]),
    17: ([0.1, 0.2, 0.2, 0.2, 0.3], [_katsuura, _ackley, _grie_rosen, _schwefel, _rastrigin]),
    18: ([0.2, 0.2, 0.2, 0.2, 0.2], [_ellips, _ackley, _rastrigin, _hgbat, _discus]),
    19: ([0.2, 0.2, 0.2, 0.2, 0.2], [_bent_cigar, _rastrigin, _grie_rosen, _weierstrass, _escaffer6]),
    20: ([0.1, 0.1, 0.2, 0.2, 0.2, 0.2], [_hgbat, _katsuura, _ackley, _rastrigin, _schwefel, _schaffer_f7]),
}
HF22 = {
    6: ([0.4, 0.4, 0.2], [_bent_cigar, _hgbat, _rastrigin]),
    7: ([0.1, 0.2, 0.2, 0.2, 0.1, 0.2], [_hgbat, _katsuura, _ackley, _rastrigin, _schwefel, _schaffer_f7]),
    8: ([0.3, 0.2, 0.2, 0.1, 0.2], [_katsuura, _happycat, _grie_rosen, _schwefel, _ackley]),
}
BASIC17 = {1: _bent_cigar, 2: _sum_diff_pow, 3: _zakharov, 4: _rosenbrock, 5: _rastrigin, 6: _schaffer_f7,
           7: _bi_rastrigin, 8: _step_rastrigin, 9: _levy, 10: _schwefel}
BASIC22 = {1: _zakharov, 2: _rosenbrock, 3: _schaffer_f7, 4: _step_rastrigin, 5: _levy}
CF17_HYB = {29: [15, 16, 17], 30: [15, 18, 19]}   # cf09 / cf10: hf05,hf06,hf07 / hf05,hf08,hf09 (CEC2017 numbering)
CF17_HYB_DB = ([10, 30, 50], [0, 100, 200])


def _scale(v, sc):
    return v if sc is None else sc[0] * v / sc[1]


def _files(suite, func, dim):
    d = os.path.join(DATA, f"cec{suite}")
    fs = [os.path.join(d, f"M_{func}_D{dim}.txt"), os.path.join(d, f"shift_data_{func}.txt")]
    if (suite == 2017 and (11 <= func <= 20 or func in (29, 30))) or (suite == 2022 and 6 <= func <= 8):
        fs.append(os.path.join(d, f"shuffle_data_{func}_D{dim}.txt"))
    return fs


def available(suite=None):
    """[(suite, func, dim)] for which the official data exist."""
    out = []
    for s in ((2017, 2022) if suite is None else (_suite(suite),)):
        for fn in range(1, 31 if s == 2017 else 13):
            for d in DIMS[s]:
                if all(os.path.exists(p) for p in _files(s, fn, d)):
                    out.append((s, fn, d))
    return out


def _suite(s):
    s = int(str(s).lower().replace("cec", ""))
    return {17: 2017, 22: 2022}.get(s, s)


class CEC:
    """One CEC function: dim, lb, ub, name, optimum, f(X) -> (n,) (X: (n, dim) or (dim,))."""

    def __init__(self, suite, func, dim):
        suite = _suite(suite)
        if suite not in DIMS:
            raise ValueError(f"unknown suite {suite}")
        if dim not in DIMS[suite]:
            raise ValueError(f"CEC{suite} is defined for D in {DIMS[suite]}")
        fs = _files(suite, func, dim)
        missing = [p for p in fs if not os.path.exists(p)]
        if missing:
            raise ValueError(f"CEC{suite} F{func} D{dim}: no official data ({os.path.basename(missing[0])})")
        self.suite, self.func, self.dim = suite, int(func), int(dim)
        self.name = f"CEC{suite}-F{func}"
        self.lb, self.ub = np.full(dim, -100.0), np.full(dim, 100.0)
        self.optimum = 100.0 * func if suite == 2017 else float(CEC22_FOPT[func - 1])
        comp = (suite == 2017 and func >= 20) or (suite == 2022 and func >= 9)
        self._kind = ("basic" if (func in (BASIC17 if suite == 2017 else BASIC22)) else
                      "hybrid" if func in (HF17 if suite == 2017 else HF22) else "comp")
        if comp:   # C reads cf_num rows of shift and cf_num matrices; only the used components matter
            if suite == 2017 and func == 20:
                k = 1
            elif suite == 2017:
                k = 3 if func in CF17_HYB else len(CF17[func][0])
            else:
                k = len(CF22[func][0])
            self.O = _rows(fs[1], k, dim)
            self.M = _vec(fs[0], k * dim * dim).reshape(k, dim, dim)
        else:
            self.O = _vec(fs[1], dim)[None, :]
            self.M = _vec(fs[0], dim * dim).reshape(1, dim, dim)
        if len(fs) > 2:
            nss = dim * (3 if (suite == 2017 and func in CF17_HYB) else 1)
            self.S = _vec(fs[2], nss, dtype=np.int64)
        else:
            self.S = None

    def f(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}), got {X.shape}")
        n, D = X.shape
        ctx = _Ctx(n, D)
        s, fn = self.suite, self.func
        if self._kind == "basic":
            g = (BASIC17 if s == 2017 else BASIC22)[fn]
            v = g(X, self.O[0], self.M[0], 1, 1, ctx, s)
        elif self._kind == "hybrid":
            Gp, funcs = (HF17 if s == 2017 else HF22)[fn]
            v = _hybrid(X, self.O[0], self.M[0], self.S, 1, 1, ctx, Gp, funcs, s)
        elif s == 2017 and fn in CF17_HYB:
            fit = []
            for i, h in enumerate(CF17_HYB[fn]):
                Gp, funcs = HF17[h]
                fit.append(_hybrid(X, self.O[i], self.M[i], self.S[i * D:(i + 1) * D], 1, 1, ctx, Gp, funcs, s))
            v = _cf_cal(X, self.O, *CF17_HYB_DB, fit)
        elif s == 2017:
            comps, delta, bias = CF17[fn]
            fit = [_scale(g(X, self.O[i], self.M[i], 1, 1, ctx, s), sc) for i, (g, sc) in enumerate(comps)]
            v = _cf_cal(X, self.O, delta, bias, fit)
        else:
            comps, delta, bias = CF22[fn]
            fit = [_scale(g(X, self.O[i], self.M[i], 1, rf, ctx, s), sc) for i, (g, sc, rf) in enumerate(comps)]
            v = _cf_cal(X, self.O, delta, bias, fit)
        return v + self.optimum

    def __call__(self, X):
        return self.f(X)

    def __repr__(self):
        return f"CEC({self.suite}, {self.func}, {self.dim})"


@lru_cache(maxsize=256)
def get(suite, func, dim) -> CEC:
    """Factory: get(2017, 5, 10), get('cec2022', 12, 20). Instances are cached (they hold only constant data)."""
    return CEC(_suite(suite), int(func), int(dim))
