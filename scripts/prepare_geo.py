"""Build the nine head and neck cancer gene expression datasets (tumor vs. non-malignant tissue).

Input (download from the Gene Expression Omnibus, https://www.ncbi.nlm.nih.gov/geo/):
    raw/<GSE>_series_matrix.txt.gz  for GSE12452, GSE53819, GSE59102, GSE143224, GSE107591, GSE30784, GSE25099,
                                    GSE37991, GSE13601
    raw/gpl/<GPL>.annot.gz          optional platform annotation, used only for gene symbols
Output: data/geo/<NAME>.npz with
    X (n, 2000) float32   log2 expression of the 2000 probes with the largest variance (label-free filter)
    y (n,) int8           1 = tumor, 0 = non-malignant tissue
    groups (n,)           patient identifier; samples of one patient are never split between training and test
    feat (2000,)          probe identifiers; sym (2000,) gene symbols (empty if no annotation)
    samples (n,)          GSM accessions
and data/geo/datasets.json with a summary of every dataset.

    python scripts/prepare_geo.py --raw path/to/raw
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "geo"
D_KEEP = 2000


def read_series_matrix(raw, gse):
    meta, rows, ids, header = {}, [], [], None
    in_table = False
    with gzip.open(raw / f"{gse}_series_matrix.txt.gz", "rt", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                p = [x.strip('"') for x in line.split("\t")]
                if header is None:
                    header = p
                    continue
                ids.append(p[0])
                rows.append([float(x) if x not in ("", "null", "NA", "NaN") else np.nan for x in p[1:]])
            elif line.startswith("!Sample_"):
                p = line.split("\t")
                meta.setdefault(p[0], []).append([x.strip('"') for x in p[1:]])
    return header[1:], np.array(ids), np.array(rows, dtype=float).T, meta


def char_field(meta, key):
    out = None
    for row in meta.get("!Sample_characteristics_ch1", []):
        vals = [v.split(": ", 1)[1] if v.lower().startswith(key.lower() + ": ") else None for v in row]
        out = vals if out is None else [a if a is not None else b for a, b in zip(out, vals)]
    return out


def annot_symbols(raw, gpl):
    f = raw / "gpl" / f"{gpl}.annot.gz"
    if not f.exists():
        return {}
    sym, header = {}, None
    with gzip.open(f, "rt", errors="replace") as fh:
        for line in fh:
            if line[0] in "#!^":
                continue
            p = line.rstrip("\n").split("\t")
            if header is None:
                header = p
                i_id, i_sym = header.index("ID"), header.index("Gene symbol")
                continue
            if len(p) > i_sym:
                sym[p[i_id]] = p[i_sym]
    return sym


def finish(raw, name, gse, gpl, X, y, groups, probes, samples, info):
    X = np.asarray(X, dtype=float)
    keep = np.all(np.isfinite(X), axis=0)
    X, probes = X[:, keep], probes[keep]
    if np.nanmax(X) > 100:                       # values on a linear scale -> log2(v + 1)
        X = np.log2(np.maximum(X, 0) + 1.0)
        info["log2_applied"] = True
    top = np.sort(np.argsort(-X.var(axis=0), kind="stable")[:D_KEEP])
    X, probes = X[:, top], probes[top]
    s = annot_symbols(raw, gpl)
    sym = np.array([s.get(p, "") for p in probes])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", X=X.astype(np.float32), y=np.asarray(y, dtype=np.int8),
                        groups=np.asarray(groups).astype(str), feat=probes.astype(str), sym=sym.astype(str),
                        samples=np.asarray(samples).astype(str))
    y = np.asarray(y)
    info.update(gse=gse, gpl=gpl, n=int(len(y)), n_tumor=int(y.sum()), n_normal=int(len(y) - y.sum()),
                n_patients=int(len(set(groups))), D=int(X.shape[1]),
                value_range=[round(float(X.min()), 3), round(float(X.max()), 3)])
    print(name, json.dumps(info))
    return name, info


def build(raw, name, gse, gpl, label_of, group_of, note):
    gsm, probes, X, meta = read_series_matrix(raw, gse)
    lab = [label_of(i, meta) for i in range(len(gsm))]
    sel = [i for i, v in enumerate(lab) if v is not None]
    return finish(raw, name, gse, gpl, X[sel], [lab[i] for i in sel], [group_of(i, meta, gsm) for i in sel],
                  probes, [gsm[i] for i in sel], dict(patients=note))


def title(meta, i):
    return meta["!Sample_title"][0][i]


def src(meta, i):
    return meta["!Sample_source_name_ch1"][0][i]


def own(i, meta, gsm):
    return gsm[i]


# (name, GEO series, platform, label rule (None = excluded), patient rule, note)
SPECS = [
    ("NPC-GSE12452", "GSE12452", "GPL570",
     lambda i, m: 1 if "carcinoma" in title(m, i) else 0, own,
     "one sample per individual"),
    ("NPC-GSE53819", "GSE53819", "GPL6480",
     lambda i, m: 1 if title(m, i).startswith("NPC PT") else 0,
     lambda i, m, g: "P" + str(int(re.search(r"-(\d+)-", title(m, i)).group(1)) % 100),
     "tumor PT-1xx and normal NCNT-2xx with the same number are treated as one patient"),
    ("LSCC-GSE59102", "GSE59102", "GPL6480",
     lambda i, m: 1 if title(m, i).startswith("Larynx") else 0,
     lambda i, m, g: re.search(r"\(\w+_(\d+)_\d+\)", title(m, i)).group(1),
     "middle field of the sample code = patient (tumor and surgical margin)"),
    ("LSCC-GSE143224", "GSE143224", "GPL5175",
     lambda i, m: 1 if "tumor" in title(m, i) else 0,
     lambda i, m, g: re.search(r"(\d+)$", title(m, i)).group(1),
     "trailing number of the title = patient"),
    ("HNSCC-GSE107591", "GSE107591", "GPL6244",
     lambda i, m: 1 if src(m, i) == "tumor" else 0,
     lambda i, m, g: re.match(r"(\d+)", title(m, i)).group(1),
     "leading number of the title = patient (paired tumor and normal)"),
    ("OSCC-GSE30784", "GSE30784", "GPL570",
     lambda i, m: {"cancer": 1, "control": 0}.get(char_field(m, "status")[i]), own,
     "one sample per individual; dysplasia samples excluded"),
    ("OSCC-GSE25099", "GSE25099", "GPL5175",
     lambda i, m: 1 if char_field(m, "disease state")[i] == "tumor" else 0, own,
     "one sample per individual"),
    ("OSCC-GSE37991", "GSE37991", "GPL6883",
     lambda i, m: 1 if title(m, i).startswith("T") else 0,
     lambda i, m, g: re.match(r"[TN](\d+)", title(m, i)).group(1),
     "T and N samples with the same number = one patient"),
    ("OSCC-GSE13601", "GSE13601", "GPL8300",
     lambda i, m: {"Tumor": 1, "Normal": 0}.get(src(m, i)),
     lambda i, m, g: re.search(r"Tongue (\d+)", title(m, i)).group(1),
     "Tongue number = patient; lymph node sample excluded"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "raw"), help="folder with the series matrix files")
    raw = Path(ap.parse_args().raw)
    info = dict(build(raw, *s) for s in SPECS)
    (OUT / "datasets.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
