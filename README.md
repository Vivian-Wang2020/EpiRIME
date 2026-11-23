# EpiRIME: Epitaxial RIME Algorithm

EpiRIME is an improved version of the RIME optimizer [1], a physics-based metaheuristic that models the growth of
rime ice. It is evaluated on the CEC2017 and CEC2022 benchmark suites and applied to wrapper gene selection on nine
public head and neck cancer gene expression datasets. The repository contains the optimizer, the benchmark functions,
the gene selection problem with the processed datasets, the scripts of all experiments, and a script that draws the
result figures of the paper from the published raw results.

## The algorithm

EpiRIME keeps the population structure, the dimension-wise soft-rime mask, and the positive greedy selection of
RIME, and replaces its two search operators by two strategies:

* **S1, droplet-accretion soft-rime search.** Every individual attaches a droplet to a *nucleus* drawn at random
  from the best individuals (the nucleus pool). The droplet is shaped by the covariance of the whole population and
  scaled by an individual *freezing intensity*, which is adapted from the successful steps of the previous
  generations. The step size therefore shrinks as the population converges and follows its principal directions.
* **S2, lattice-oriented puncture.** The puncture copies coordinates of the best individual in the coordinate
  system given by the principal axes of the elite individuals (the *lattice frame*), with a probability that
  depends on the fitness rank of the individual. The puncture operator is thus invariant to rotations of the search
  space and to strictly increasing transformations of the objective.

The only new parameter is the nucleus-pool ratio `rho` (default 0.3). The update equations are given in the
docstring of [`epirime/optimizer.py`](epirime/optimizer.py) and in the paper.

## Repository contents

```
epirime/
    optimizer.py          EpiRIME
    cec.py                CEC2017 and CEC2022 benchmark functions (NumPy port of the official C code)
    cec_data/             official shift, rotation, and shuffle data (CEC2017: D = 30, 50; CEC2022: D = 10, 20)
    gene_selection.py     gene selection problem: fitness, patient-level data split, test error
data/geo/                 the nine processed gene expression datasets (.npz) and a summary (datasets.json)
scripts/
    prepare_geo.py        rebuilds data/geo/ from the GEO series matrix files
    plot_results.py       draws the result figures of the paper from the results package
run_cec.py                benchmark experiments
run_gene_selection.py     gene selection experiments
requirements.txt
LICENSE
```

## Installation

Python 3.8 or later with NumPy is required; matplotlib is needed only for `scripts/plot_results.py`.

```
git clone <URL of this repository>
cd EpiRIME
pip install -r requirements.txt
```

The code is not installed as a package; run the scripts from the repository folder, or add the folder to
`PYTHONPATH` to import `epirime` elsewhere.

## Quick start

EpiRIME minimizes a *vectorized* objective function that maps an array of shape `(n, D)` (one candidate solution
per row) to an array of `n` objective values. The bounds `lb` and `ub` are scalars or arrays of length `D`.

```python
import numpy as np
from epirime import EpiRIME

def sphere(X):
    return np.sum(X ** 2, axis=1)

opt = EpiRIME(pop_size=100, max_fes=20_000, seed=1)
res = opt.optimize(sphere, -100, 100, 20)        # objective, lb, ub, D
print(res.best_fit)                               # best objective value
print(res.best_x)                                 # best solution
```

| Argument   | Default | Meaning                                                                    |
|------------|---------|----------------------------------------------------------------------------|
| `pop_size` | 100     | population size N                                                          |
| `max_fes`  | 300000  | budget of objective function evaluations; the run stops when it is used up |
| `seed`     | None    | seed of NumPy's random generator                                           |
| `rho`      | 0.3     | nucleus-pool ratio: the nucleus pool holds the best round(rho N) individuals |

`optimize` returns a `Result` with the fields `best_fit`, `best_x`, `fes` (number of evaluations used), and the
convergence record `curve_fes` / `curve_fit` (best objective value after each batch of evaluations).

Each generation performs two eigen-decompositions of D x D covariance matrices, so the cost per generation grows
as O(D^3). For D up to a few hundred this is negligible compared with most objective functions; for D = 2000 (gene
selection) a run of 20,000 evaluations takes about 13 minutes on one CPU core.

### Benchmark functions

```python
from epirime.cec import get

f = get(2017, 5, 30)          # suite, function number, dimension
values = f(X)                 # X of shape (n, 30)
f.lb, f.ub, f.optimum         # bounds and optimal value
```

The functions reproduce the official C implementations of the CEC2017 [2] and CEC2022 [3] competitions on
bound-constrained single-objective optimization, including their known particularities (listed in the docstring of
[`epirime/cec.py`](epirime/cec.py)). CEC2017 F2 was removed from the suite by the organizers and is not used. Only
the data for the dimensions of the paper are included (CEC2017: D = 30, 50; CEC2022: D = 10, 20).

## Reproducing the experiments

### Benchmark experiments

Settings of the paper: budget 1000 D function evaluations, population size 100, `rho = 0.3`, 30 independent runs
per function; CEC2017 F1 and F3 to F30 with D = 30 and 50, CEC2022 F1 to F12 with D = 10 and 20.

```
python run_cec.py --suite 2017 --dim 30 --jobs 8 --out cec2017_d30.json
python run_cec.py --suite 2017 --dim 50 --jobs 8 --out cec2017_d50.json
python run_cec.py --suite 2022 --dim 10 --jobs 8 --out cec2022_d10.json
python run_cec.py --suite 2022 --dim 20 --jobs 8 --out cec2022_d20.json
```

| Option    | Default     | Meaning                                                   |
|-----------|-------------|-----------------------------------------------------------|
| `--suite` | (required)  | 2017 or 2022                                              |
| `--dim`   | (required)  | 30 or 50 for CEC2017, 10 or 20 for CEC2022                |
| `--funcs` | whole suite | function numbers, e.g. `--funcs 1 6 11`                   |
| `--runs`  | 30          | number of runs per function                               |
| `--rho`   | 0.3         | nucleus-pool ratio (the sensitivity analysis varies it)   |
| `--jobs`  | 1           | number of worker processes                                |
| `--out`   | none        | JSON file for the results                                 |

The script prints the mean and standard deviation of the best objective value of every function. The JSON file
holds `fitness[function][run]` (best objective value; the optimum is not subtracted) and `curves[function][run]`
(best objective value at 50 equally spaced points of the budget). One run takes about 0.5 s (CEC2017, D = 30) to
1.5 s (D = 50) on one CPU core.

### Gene selection experiments

Each dataset is a binary classification task, tumor against non-malignant tissue, restricted to the 2000 probes with
the largest variance. A candidate solution `x` in `[0, 1]^2000` selects gene `j` when `x_j > 0.5`, and the fitness
to be minimized is

```
F(x) = 0.99 BE(S) + 0.01 |S| / 2000,
```

where `BE(S)` is the leave-one-patient-out balanced error of a k-nearest-neighbor classifier (k = 3) on the
training samples and `|S|` is the number of selected genes. In each run, the patients are split into 70% training
and 30% test data, stratified by class, so that all samples of one patient fall on the same side. After the
search, the selected genes are evaluated by the balanced error on the held-out test patients.

Settings of the paper: budget 20,000 fitness evaluations, population size 100, `rho = 0.3`, 30 runs per dataset.

```
python run_gene_selection.py --jobs 8 --out gene_selection.json          # all datasets, 30 runs each
python run_gene_selection.py --datasets NPC-GSE53819 --runs 3            # a short test
```

| Option       | Default      | Meaning                                    |
|--------------|--------------|--------------------------------------------|
| `--datasets` | all nine     | dataset names (see the table below)        |
| `--runs`     | 30           | number of runs per dataset                 |
| `--fes`      | 20000        | budget of fitness evaluations              |
| `--rho`      | 0.3          | nucleus-pool ratio                         |
| `--jobs`     | 1            | number of worker processes                 |
| `--out`      | none         | JSON file for the results                  |

The script prints, for every dataset, the mean training fitness, number of selected genes, training balanced
error, and test balanced error. The JSON file lists every run with these values and the probe identifiers of the
selected genes. All 270 runs take about 60 CPU hours.

The problem can also be used directly:

```python
from epirime import EpiRIME
from epirime.gene_selection import GeneSelection

p = GeneSelection("NPC-GSE53819", run=0)          # dataset and run (the run fixes the data split)
res = EpiRIME(pop_size=100, max_fes=20_000, seed=1).optimize(p.f, p.lb, p.ub, p.dim)
m = p.mask(res.best_x)                            # boolean mask of the selected genes
print(m.sum(), p.train_error(m), p.test_error(res.best_x), p.probes[m])
```

### Seeds

All scripts use the seeds of the paper, so the runs are paired across algorithms:

* benchmark functions: run r on function F uses the seed 2026 + 7 F + 20 r;
* gene selection: run r on dataset number k (order of `DATASETS` in `epirime/gene_selection.py`, starting at 0)
  uses the data split seed 7,000,000 + 1000 k + r and the optimizer seed 8,000,000 + 1000 k + r.

The results of the paper were obtained with NumPy 2.4. Other NumPy versions or linear algebra libraries can change
the results in the last digits.

## Results and figures

The raw results of all experiments of the paper are published as a separate package, `EpiRIME-results.zip`,
attached to the [Releases](../../releases) of this repository. It contains, for the benchmark experiments (twelve
algorithms), the ablation analysis, the sensitivity analysis, and the gene selection experiments (nine algorithms),
the result of every run, the convergence records, the CPU times, and the selected genes. Its README describes the
file formats. All tables and figures of the paper and its supplementary material can be computed from these files.

The result figures of the paper (Figures 3 to 9) and of its supplementary material (Figures S1 to S24) are drawn
with

```
python scripts/plot_results.py --results path/to/EpiRIME-results --out figures
```

which writes every figure as PDF and PNG (`--only main`, `--only supplementary`, or `--only gene_selection` draws a
part of them). The list of figures is given at the top of [`scripts/plot_results.py`](scripts/plot_results.py).

## Data

The gene expression data are public series of the Gene Expression Omnibus (GEO). `data/geo/` holds the processed
datasets: log2 expression values of the 2000 probes with the largest variance (a label-free filter), the class
labels, the patient identifiers used for the patient-level split, the probe identifiers and gene symbols, and the
GSM accessions of the samples.

| Name            | Site                          | GEO series | Platform | Samples (tumor/normal) | Patients |
|-----------------|-------------------------------|------------|----------|------------------------|----------|
| NPC-GSE12452    | nasopharynx                   | GSE12452   | GPL570   | 41 (31/10)             | 41       |
| NPC-GSE53819    | nasopharynx                   | GSE53819   | GPL6480  | 36 (18/18)             | 18       |
| LSCC-GSE59102   | larynx                        | GSE59102   | GPL6480  | 42 (29/13)             | 31       |
| LSCC-GSE143224  | larynx                        | GSE143224  | GPL5175  | 25 (14/11)             | 14       |
| HNSCC-GSE107591 | oral cavity, larynx, pharynx  | GSE107591  | GPL6244  | 47 (24/23)             | 24       |
| OSCC-GSE30784   | oral cavity                   | GSE30784   | GPL570   | 212 (167/45)           | 212      |
| OSCC-GSE25099   | oral cavity                   | GSE25099   | GPL5175  | 79 (57/22)             | 79       |
| OSCC-GSE37991   | oral cavity                   | GSE37991   | GPL6883  | 80 (40/40)             | 40       |
| OSCC-GSE13601   | oral tongue                   | GSE13601   | GPL8300  | 57 (31/26)             | 37       |

NPC: nasopharyngeal carcinoma; LSCC: laryngeal squamous cell carcinoma; HNSCC: head and neck squamous cell
carcinoma; OSCC: oral squamous cell carcinoma. `data/geo/datasets.json` records how the patients were identified
in each series.

To rebuild `data/geo/` from the original files, download the series matrix files (`<GSE>_series_matrix.txt.gz`)
from GEO into a folder, optionally with the platform annotations (`gpl/<GPL>.annot.gz`, used only for the gene
symbols), and run

```
python scripts/prepare_geo.py --raw path/to/folder
```

The benchmark data in `epirime/cec_data/` are the official data files of the CEC2017 and CEC2022 competitions.

## Citation

The reference of the paper will be added here after publication.

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The gene expression data are derived from
public GEO series and the benchmark data from the CEC competitions; please cite the original sources when you use
them.

## References

[1] H. Su, D. Zhao, A. A. Heidari, L. Liu, X. Zhang, M. Mafarja, H. Chen. RIME: A physics-based optimization.
Neurocomputing 532 (2023) 183-214. https://doi.org/10.1016/j.neucom.2023.02.010

[2] N. H. Awad, M. Z. Ali, J. J. Liang, B. Y. Qu, P. N. Suganthan. Problem definitions and evaluation criteria for
the CEC 2017 special session and competition on single objective bound constrained real-parameter numerical
optimization. Technical report, Nanyang Technological University, Singapore, 2016.

[3] A. Kumar, K. V. Price, A. W. Mohamed, A. A. Hadi, P. N. Suganthan. Problem definitions and evaluation criteria
for the CEC 2022 special session and competition on single objective bound constrained numerical optimization.
Technical report, Nanyang Technological University, Singapore, 2021.
