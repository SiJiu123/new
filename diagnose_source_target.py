import os
from typing import Dict, List, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from run_demo2 import (
    TEST_BULK_FILE,
    TRAIN_CELLTYPES_FILE,
    TRAIN_COUNTS_FILE,
    TYPE_LIST,
    load_test_bulk_from_302c,
    load_train_adata_from_296c,
)


# =====================
# Config
# =====================
# If source_data / target_data already exist in memory,
# you can import and call diagnose_adata(source_data, target_data) directly.
# By default, this script reuses run_demo2.py inputs.
OUT_DIR = r"E:/Desktop/new/diagnostics"

# Major-alert thresholds
SCALE_ALERT_RATIO = 10.0
GENE_OVERLAP_ALERT_RATIO = 0.95


def _banner(msg: str) -> None:
    print("\n" + "=" * 88)
    print(msg)
    print("=" * 88)


def _warn(msg: str) -> None:
    print(f"\n[!!! WARNING !!!] {msg}")


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _basic_stats(x) -> Dict[str, float]:
    x_dense = _to_dense(x).astype(np.float64, copy=False)
    return {
        "min": float(np.min(x_dense)),
        "max": float(np.max(x_dense)),
        "mean": float(np.mean(x_dense)),
        "var": float(np.var(x_dense)),
    }


def _nonzero_ratio(x) -> float:
    if sparse.issparse(x):
        return float(x.nnz / (x.shape[0] * x.shape[1]))
    arr = np.asarray(x)
    return float(np.count_nonzero(arr) / arr.size)


def _mean_expression_by_gene(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel()
    arr = np.asarray(x)
    return arr.mean(axis=0)


def _format_stats(title: str, stats: Dict[str, float]) -> str:
    return (
        f"{title}: min={stats['min']:.6g}, max={stats['max']:.6g}, "
        f"mean={stats['mean']:.6g}, var={stats['var']:.6g}"
    )


def check_gene_alignment(source_data: ad.AnnData, target_data: ad.AnnData) -> Tuple[pd.DataFrame, Dict[str, float]]:
    _banner("1) Gene Alignment")

    source_genes = source_data.var_names.astype(str)
    target_genes = target_data.var_names.astype(str)

    n_source = len(source_genes)
    n_target = len(target_genes)
    same_count = n_source == n_target
    same_order = same_count and source_genes.equals(target_genes)

    source_set = set(source_genes)
    target_set = set(target_genes)
    overlap = source_set.intersection(target_set)

    overlap_ratio_source = len(overlap) / n_source if n_source > 0 else 0.0
    overlap_ratio_target = len(overlap) / n_target if n_target > 0 else 0.0

    print(f"source genes: {n_source}")
    print(f"target genes: {n_target}")
    print(f"same gene count: {same_count}")
    print(f"same exact order: {same_order}")
    print(f"overlap genes: {len(overlap)}")
    print(f"overlap ratio (vs source): {overlap_ratio_source:.4f}")
    print(f"overlap ratio (vs target): {overlap_ratio_target:.4f}")

    if not same_order:
        _warn("Gene order is NOT identical between source and target.")
    else:
        _ok("Gene order is identical.")

    if overlap_ratio_source < GENE_OVERLAP_ALERT_RATIO or overlap_ratio_target < GENE_OVERLAP_ALERT_RATIO:
        _warn(
            f"Gene overlap ratio is below {GENE_OVERLAP_ALERT_RATIO:.2f}. "
            "This is a major risk for deconvolution performance."
        )
    else:
        _ok("Gene overlap ratio is high.")

    # Build aligned mean-expression frame on overlap genes for later plotting.
    overlap_sorted = sorted(overlap)
    source_mean = pd.Series(_mean_expression_by_gene(source_data.X), index=source_genes)
    target_mean = pd.Series(_mean_expression_by_gene(target_data.X), index=target_genes)

    aligned_df = pd.DataFrame(
        {
            "source_mean": source_mean.reindex(overlap_sorted).to_numpy(),
            "target_mean": target_mean.reindex(overlap_sorted).to_numpy(),
        },
        index=overlap_sorted,
    ).dropna()

    summary = {
        "n_source": float(n_source),
        "n_target": float(n_target),
        "same_count": float(same_count),
        "same_order": float(same_order),
        "overlap": float(len(overlap)),
        "overlap_ratio_source": float(overlap_ratio_source),
        "overlap_ratio_target": float(overlap_ratio_target),
    }
    return aligned_df, summary


def check_distribution_and_scale(source_data: ad.AnnData, target_data: ad.AnnData) -> Dict[str, float]:
    _banner("2) Distribution & Scale")

    src_stats = _basic_stats(source_data.X)
    tgt_stats = _basic_stats(target_data.X)

    print(_format_stats("source", src_stats))
    print(_format_stats("target", tgt_stats))

    # Use robust high-end scale (99th percentile) to avoid outlier domination.
    src_dense = _to_dense(source_data.X).astype(np.float64, copy=False)
    tgt_dense = _to_dense(target_data.X).astype(np.float64, copy=False)
    src_p99 = float(np.percentile(src_dense, 99))
    tgt_p99 = float(np.percentile(tgt_dense, 99))

    ratio = np.inf if min(src_p99, tgt_p99) == 0 and max(src_p99, tgt_p99) > 0 else (
        max(src_p99, tgt_p99) / max(min(src_p99, tgt_p99), 1e-12)
    )

    print(f"source p99: {src_p99:.6g}")
    print(f"target p99: {tgt_p99:.6g}")
    print(f"p99 scale ratio (larger/smaller): {ratio:.4g}")

    if ratio >= SCALE_ALERT_RATIO:
        _warn(
            f"Detected major scale mismatch (>= {SCALE_ALERT_RATIO:.1f}x). "
            "Possible preprocessing inconsistency (raw count vs log-normalized)."
        )
    else:
        _ok("No extreme scale mismatch by p99 ratio.")

    # Heuristic hint for log-like range.
    if src_stats["max"] <= 20 and tgt_stats["max"] > 100:
        _warn("source looks log-like (max <= 20), target looks count-like (max > 100).")
    if tgt_stats["max"] <= 20 and src_stats["max"] > 100:
        _warn("target looks log-like (max <= 20), source looks count-like (max > 100).")

    return {
        "source_min": src_stats["min"],
        "source_max": src_stats["max"],
        "source_mean": src_stats["mean"],
        "source_var": src_stats["var"],
        "target_min": tgt_stats["min"],
        "target_max": tgt_stats["max"],
        "target_mean": tgt_stats["mean"],
        "target_var": tgt_stats["var"],
        "source_p99": src_p99,
        "target_p99": tgt_p99,
        "p99_ratio": float(ratio),
    }


def plot_mean_expression_correlation(aligned_mean_df: pd.DataFrame, out_dir: str) -> Dict[str, float]:
    _banner("3) Mean Expression Correlation")

    os.makedirs(out_dir, exist_ok=True)
    if aligned_mean_df.empty:
        _warn("Aligned mean-expression table is empty. Skip plotting.")
        return {"pearson": np.nan, "n_genes": 0.0}

    x = aligned_mean_df["source_mean"].to_numpy()
    y = aligned_mean_df["target_mean"].to_numpy()

    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan
    print(f"overlap genes used for correlation: {len(x)}")
    print(f"pearson(source_mean, target_mean): {pearson:.6f}")

    # Raw scale scatter
    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, s=8, alpha=0.45)
    plt.xlabel("Source mean expression")
    plt.ylabel("Target mean expression")
    plt.title(f"Mean Expression Correlation (raw)\nPearson={pearson:.4f}")
    raw_fig = os.path.join(out_dir, "mean_expr_corr_raw.png")
    plt.tight_layout()
    plt.savefig(raw_fig, dpi=150)
    plt.close()

    # log1p scale scatter for better dynamic-range visibility
    lx = np.log1p(x)
    ly = np.log1p(y)
    lpearson = float(np.corrcoef(lx, ly)[0, 1]) if len(lx) > 1 else np.nan

    plt.figure(figsize=(7, 6))
    plt.scatter(lx, ly, s=8, alpha=0.45)
    plt.xlabel("log1p(Source mean expression)")
    plt.ylabel("log1p(Target mean expression)")
    plt.title(f"Mean Expression Correlation (log1p)\nPearson={lpearson:.4f}")
    log_fig = os.path.join(out_dir, "mean_expr_corr_log1p.png")
    plt.tight_layout()
    plt.savefig(log_fig, dpi=150)
    plt.close()

    print(f"saved plot: {raw_fig}")
    print(f"saved plot: {log_fig}")

    if np.isfinite(pearson) and pearson < 0.5:
        _warn("Low mean-expression correlation (< 0.5). Domain shift risk is high.")

    return {"pearson": pearson, "pearson_log1p": lpearson, "n_genes": float(len(x))}


def check_sparsity(source_data: ad.AnnData, target_data: ad.AnnData) -> Dict[str, float]:
    _banner("4) Sparsity")

    src_nonzero = _nonzero_ratio(source_data.X)
    tgt_nonzero = _nonzero_ratio(target_data.X)

    print(f"source nonzero ratio: {src_nonzero:.6f} (sparsity={1 - src_nonzero:.6f})")
    print(f"target nonzero ratio: {tgt_nonzero:.6f} (sparsity={1 - tgt_nonzero:.6f})")

    dense_ratio = (max(src_nonzero, tgt_nonzero) / max(min(src_nonzero, tgt_nonzero), 1e-12))
    if dense_ratio >= SCALE_ALERT_RATIO:
        _warn("Nonzero-ratio gap is huge (>= 10x), indicating major sparsity-pattern shift.")

    return {
        "source_nonzero_ratio": src_nonzero,
        "target_nonzero_ratio": tgt_nonzero,
        "nonzero_ratio_gap": float(dense_ratio),
    }


def check_label_consistency(source_data: ad.AnnData, target_data: ad.AnnData) -> Dict[str, float]:
    _banner("5) Label Consistency")

    if "cell_types" not in source_data.uns:
        _warn("source_data.uns['cell_types'] not found.")
        return {
            "n_source_cell_types": 0.0,
            "n_missing_in_target_obs": np.nan,
            "order_match": np.nan,
        }

    source_cell_types: List[str] = [str(x) for x in source_data.uns["cell_types"]]
    target_obs_cols = list(map(str, target_data.obs.columns))

    missing = [c for c in source_cell_types if c not in target_obs_cols]
    present = [c for c in source_cell_types if c in target_obs_cols]

    print(f"source cell_types (ordered): {source_cell_types}")
    print(f"target obs columns count: {len(target_obs_cols)}")
    print(f"present label columns in target.obs: {len(present)}/{len(source_cell_types)}")
    print(f"missing label columns: {missing}")

    # Order check among present labels.
    target_filtered = [c for c in target_obs_cols if c in source_cell_types]
    source_filtered = [c for c in source_cell_types if c in target_obs_cols]
    order_match = target_filtered == source_filtered and len(source_filtered) == len(source_cell_types)

    print(f"label order match for model output dims: {order_match}")

    if missing:
        _warn("Some source cell types are missing in target.obs. Metric/eval mapping may be wrong.")
    else:
        _ok("All source cell types exist in target.obs.")

    if not order_match:
        _warn("Cell type order mismatch detected. Ensure output dim mapping follows source_data.uns['cell_types'] exactly.")
    else:
        _ok("Cell type order is consistent.")

    return {
        "n_source_cell_types": float(len(source_cell_types)),
        "n_missing_in_target_obs": float(len(missing)),
        "order_match": float(order_match),
    }


def diagnose_adata(source_data: ad.AnnData, target_data: ad.AnnData, out_dir: str = OUT_DIR) -> pd.DataFrame:
    _banner("Deconvolution Data Diagnostic")
    print(f"source shape: {source_data.shape}")
    print(f"target shape: {target_data.shape}")

    aligned_mean_df, gene_info = check_gene_alignment(source_data, target_data)
    dist_info = check_distribution_and_scale(source_data, target_data)
    corr_info = plot_mean_expression_correlation(aligned_mean_df, out_dir=out_dir)
    sparsity_info = check_sparsity(source_data, target_data)
    label_info = check_label_consistency(source_data, target_data)

    summary = {}
    summary.update(gene_info)
    summary.update(dist_info)
    summary.update(corr_info)
    summary.update(sparsity_info)
    summary.update(label_info)

    summary_df = pd.DataFrame([summary])
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "source_target_diagnosis_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    _banner("Diagnosis Summary Saved")
    print(f"summary csv: {summary_path}")
    return summary_df


def load_adata_from_run_demo2_inputs() -> Tuple[ad.AnnData, ad.AnnData]:
    _banner("Loading source/target from run_demo2 inputs")
    print(f"TRAIN_COUNTS_FILE: {TRAIN_COUNTS_FILE}")
    print(f"TRAIN_CELLTYPES_FILE: {TRAIN_CELLTYPES_FILE}")
    print(f"TEST_BULK_FILE: {TEST_BULK_FILE}")
    print(f"TYPE_LIST: {TYPE_LIST}")

    src = load_train_adata_from_296c(TRAIN_COUNTS_FILE, TRAIN_CELLTYPES_FILE, TYPE_LIST)
    src.uns["cell_types"] = list(TYPE_LIST)
    tgt = load_test_bulk_from_302c(TEST_BULK_FILE, src.var_names, TYPE_LIST)
    return src, tgt


if __name__ == "__main__":
    src, tgt = load_adata_from_run_demo2_inputs()
    diagnose_adata(src, tgt, out_dir=OUT_DIR)