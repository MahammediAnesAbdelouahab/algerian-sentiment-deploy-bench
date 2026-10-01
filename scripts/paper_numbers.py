"""Derive the numbers quoted in the paper (ratios, ranges, gaps) from the result tables.

Run from the repository root: python scripts/paper_numbers.py  (writes results/paper_numbers.json)
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "tables") + os.sep
A = os.path.join(ROOT, "data", "audit") + os.sep
Q = pd.read_csv(R + "table_quality.csv")
D = pd.read_csv(R + "table_deployment.csv")
SC = pd.read_csv(R + "table_twifil_shortcut.csv")
CI = pd.read_csv(R + "table_bootstrap_ci.csv")
ST = pd.read_csv(A + "dataset_stats.csv")
AU = pd.read_csv(A + "cleaning_audit.csv")
PROV = json.load(open(os.path.join(ROOT, "results", "provenance", "provenance_02_benchmark.json")))
F = {}


def mean_sd(s):
    s = str(s)
    if "±" in s:
        m, sd = s.split("±")
        return float(m), float(sd)
    return float(s), None


# ---------------- quality ----------------
for r in Q.itertuples(index=False):
    m, sd = mean_sd(r[4])
    a, asd = mean_sd(r[5])
    F[f"q|{r.dataset}|{r.model}"] = {"f1": m, "f1_sd": sd, "acc": a, "acc_sd": asd, "seeds": int(r.seeds)}


def q(ds, m):
    return F[f"q|{ds}|{m}"]


best_cls = {}
for ds in ["twifil_clean", "youtube", "twifil_official", "narabizi"]:
    cl = {m: q(ds, m)["f1"] for m in ["NB-char", "SVM-char", "LR-word+char"]}
    tr = {m: q(ds, m)["f1"] for m in ["DziriBERT", "MARBERTv2", "CAMeLBERT-DA", "mE5-small"]}
    bc, bt = max(cl, key=cl.get), max(tr, key=tr.get)
    best_cls[ds] = bc
    F[f"gap_seedmean|{ds}"] = {"best_classical": bc, "best_transformer": bt,
                               "gap": round(tr[bt] - cl[bc], 1),
                               "transformers_below_best_classical": [m for m, v in tr.items() if v < cl[bc]]}

# ---------------- bootstrap ----------------
for r in CI.itertuples(index=False):
    F[f"ci|{r.dataset}|{r.model}"] = {"f1": r[2], "ci": r[3]}

# ---------------- shortcut ----------------
for r in SC.itertuples(index=False):
    d = {"no_text": r[2], "text": r[3], "overall": r[4]}
    if not np.isnan(r[4]):
        d["inflation"] = round(r[4] - r[3], 1)
    F[f"shortcut|{r[0]}"] = d

# ---------------- deployment ----------------
D["key"] = D["dataset"] + "|" + D["model"] + "|" + D["format"]
dep = D.set_index("key")


def g(ds, m, fmt, col):
    return float(dep.loc[f"{ds}|{m}|{fmt}", col])


TR = ["DziriBERT", "MARBERTv2", "CAMeLBERT-DA", "mE5-small"]
for ds in ["twifil_clean", "youtube"]:
    for m in TR:
        pt = {c: g(ds, m, "PyTorch fp32", c) for c in ["size (MB)", "peak RSS (MB)", "cold start (s)", "p50 (ms)", "p95 (ms)", "macro-F1 (%)"]}
        for fmt in ["ONNX fp32", "ONNX int8", "ONNX int8 per-channel+RR", "ONNX int8 per-channel (no RR)"]:
            x = {c: g(ds, m, fmt, c) for c in pt}
            F[f"ratio|{ds}|{m}|{fmt}"] = {
                "speedup_p50": round(pt["p50 (ms)"] / x["p50 (ms)"], 2),
                "size_ratio": round(pt["size (MB)"] / x["size (MB)"], 2),
                "rss_ratio": round(pt["peak RSS (MB)"] / x["peak RSS (MB)"], 2),
                "cold_ratio": round(pt["cold start (s)"] / x["cold start (s)"], 2),
                "delta_f1": round(x["macro-F1 (%)"] - pt["macro-F1 (%)"], 2),
                **{k: x[k] for k in x},
            }
    # SVM-char vs best transformer int8 configs
    bc = best_cls[ds]
    c = {k: g(ds, bc, "scikit-learn", k) for k in ["size (MB)", "peak RSS (MB)", "cold start (s)", "p50 (ms)", "p95 (ms)", "macro-F1 (%)"]}
    F[f"classical_artefact|{ds}|{bc}"] = c
    for fmt in ["ONNX int8", "ONNX int8 per-channel+RR"]:
        t = {k: g(ds, "DziriBERT", fmt, k) for k in c}
        F[f"cls_vs_dziri|{ds}|{fmt}"] = {
            "latency_ratio": round(t["p50 (ms)"] / c["p50 (ms)"], 1),
            "size_ratio": round(t["size (MB)"] / c["size (MB)"], 1),
            "rss_ratio": round(t["peak RSS (MB)"] / c["peak RSS (MB)"], 2),
            "f1_gap": round(t["macro-F1 (%)"] - c["macro-F1 (%)"], 1)}
    # ranges over the 4 transformers
    for fmt in ["ONNX fp32", "ONNX int8", "ONNX int8 per-channel+RR"]:
        sp = [F[f"ratio|{ds}|{m}|{fmt}"]["speedup_p50"] for m in TR]
        F[f"range_speedup|{ds}|{fmt}"] = [min(sp), max(sp)]
    for k in ["size_ratio", "rss_ratio", "cold_ratio"]:
        v = [F[f"ratio|{ds}|{m}|ONNX int8 per-channel+RR"][k] for m in TR]
        F[f"range_{k}|{ds}|int8RR"] = [min(v), max(v)]
    # p95 under 100 ms?
    sub = D[(D.dataset == ds) & (~D.format.str.contains("no RR"))]
    F[f"p95_under_100|{ds}"] = sub[sub["p95 (ms)"] <= 100][["model", "format", "p95 (ms)", "size (MB)", "macro-F1 (%)"]].values.tolist()
    F[f"min_transformer_p95|{ds}"] = float(sub[sub.model.isin(TR)]["p95 (ms)"].min())
    F[f"timing_spread_range|{ds}"] = [float(sub["p50 spread across passes (%)"].min()), float(sub["p50 spread across passes (%)"].max())]
    F[f"rss_range_sklearn|{ds}"] = [float(sub[sub.format == "scikit-learn"]["peak RSS (MB)"].min()), float(sub[sub.format == "scikit-learn"]["peak RSS (MB)"].max())]
    F[f"rss_range_pytorch|{ds}"] = [float(sub[sub.format == "PyTorch fp32"]["peak RSS (MB)"].min()), float(sub[sub.format == "PyTorch fp32"]["peak RSS (MB)"].max())]
    F[f"cold_range_pytorch|{ds}"] = [float(sub[sub.format == "PyTorch fp32"]["cold start (s)"].min()), float(sub[sub.format == "PyTorch fp32"]["cold start (s)"].max())]
    F[f"p50_range_pytorch|{ds}"] = [float(sub[sub.format == "PyTorch fp32"]["p50 (ms)"].min()), float(sub[sub.format == "PyTorch fp32"]["p50 (ms)"].max())]

# ---------------- data ----------------
st = ST.set_index(["dataset", "split"])
for ds in ["twifil_official", "twifil_clean", "youtube", "narabizi", "narabizi_4class"]:
    for sp in ["train", "dev", "test", "all"]:
        F[f"stats|{ds}|{sp}"] = {k: (None if pd.isna(v) else v) for k, v in st.loc[(ds, sp)].to_dict().items()}
au = AU.set_index(["dataset", "step"])
F["audit"] = {f"{d}|{s}": {k: int(v) for k, v in row.items()} for (d, s), row in au.iterrows()}

F["prov"] = {k: PROV.get(k) for k in ["bench_cpu", "cpu_flags", "versions", "runs_done", "artefacts_benchmarked"]}
F["prov"]["config"] = {k: PROV["config"][k] for k in ["SEEDS", "EPOCHS", "PATIENCE", "BATCH_SIZE", "MAX_LEN_CAP",
                                                       "CLASSICAL_GRIDS", "BENCH_THREADS", "N_SINGLE", "BENCH_ROUNDS",
                                                       "BENCH_BATCH", "TIMING_REPEATS", "N_TIMING"]}
json.dump(F, open(os.path.join(ROOT, "results", "paper_numbers.json"), "w"), indent=1, ensure_ascii=False, default=float)

# ---------- compact printout ----------
for k, v in F.items():
    if k.startswith(("gap_", "cls_vs", "range_", "p95_under", "min_transformer", "timing_spread", "rss_range", "cold_range", "p50_range", "shortcut")):
        print(k, v)
for ds in ["twifil_clean", "youtube"]:
    for m in TR:
        for fmt in ["ONNX int8", "ONNX int8 per-channel+RR", "ONNX int8 per-channel (no RR)", "ONNX fp32"]:
            r = F[f"ratio|{ds}|{m}|{fmt}"]
            print(f"{ds:12s} {m:12s} {fmt:30s} dF1 {r['delta_f1']:+6.2f} speedup {r['speedup_p50']:4.2f} size÷{r['size_ratio']:4.2f} rss÷{r['rss_ratio']:4.2f} cold÷{r['cold_ratio']:5.2f}")
print(F["prov"])
