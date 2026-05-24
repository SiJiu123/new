import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
import torch.utils.data as Data


def _ensure_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def _normalize_rows(y: np.ndarray) -> np.ndarray:
    row_sum = np.sum(y, axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return y / row_sum


def _entropy_rows(y: np.ndarray) -> np.ndarray:
    y_norm = _normalize_rows(y)
    eps = 1e-12
    entropy = -np.sum(y_norm * np.log(y_norm + eps), axis=1)
    if y_norm.shape[1] > 1:
        entropy = entropy / np.log(y_norm.shape[1])
    return entropy


def _dominant_margin(y: np.ndarray) -> np.ndarray:
    if y.shape[1] < 2:
        return np.zeros(y.shape[0], dtype=np.float32)
    sorted_y = np.sort(y, axis=1)
    return sorted_y[:, -1] - sorted_y[:, -2]


def _local_smoothness_proxy(embeddings: np.ndarray, labels: np.ndarray, k_neighbors: int) -> Dict[str, np.ndarray]:
    n_samples = embeddings.shape[0]
    if n_samples <= 1:
        zeros = np.zeros(n_samples, dtype=np.float32)
        return {
            "local_label_l1": zeros,
            "local_emb_dist": zeros,
            "local_variation_ratio": zeros,
        }

    k_query = min(k_neighbors + 1, n_samples)
    tree = cKDTree(embeddings)
    distances, indices = tree.query(embeddings, k=k_query)

    if k_query == 1:
        local_label_l1 = np.zeros(n_samples, dtype=np.float32)
        local_emb_dist = np.zeros(n_samples, dtype=np.float32)
        local_variation_ratio = np.zeros(n_samples, dtype=np.float32)
        return {
            "local_label_l1": local_label_l1,
            "local_emb_dist": local_emb_dist,
            "local_variation_ratio": local_variation_ratio,
        }

    neighbor_indices = indices[:, 1:]
    neighbor_distances = distances[:, 1:]

    label_diffs = np.abs(labels[:, None, :] - labels[neighbor_indices])
    local_label_l1 = label_diffs.mean(axis=(1, 2)).astype(np.float32)
    local_emb_dist = neighbor_distances.mean(axis=1).astype(np.float32)
    local_variation_ratio = (local_label_l1 / (local_emb_dist + 1e-8)).astype(np.float32)

    return {
        "local_label_l1": local_label_l1,
        "local_emb_dist": local_emb_dist,
        "local_variation_ratio": local_variation_ratio,
    }


def _build_loader(source_data, labels: List[str], batch_size: int) -> Data.DataLoader:
    source_x = _ensure_dense(source_data.X)
    source_ratios = [source_data.obs[ctype] for ctype in labels]
    source_y = np.asarray(source_ratios, dtype=np.float32).transpose()

    tr_data = torch.FloatTensor(source_x)
    tr_labels = torch.FloatTensor(source_y)
    source_dataset = Data.TensorDataset(tr_data, tr_labels)
    return Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=False)


def compute_camp_frequency_profile(
    model_da,
    source_data,
    batch_size: int,
    labels: List[str],
    mixed_margin: float = 0.25,
    k_neighbors: int = 5,
    out_dir: Optional[str] = None,
) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Camp frequency probe expects a CUDA-enabled environment.")

    os.makedirs(out_dir, exist_ok=True) if out_dir is not None else None

    loader = _build_loader(source_data, labels, batch_size)
    model_da.encoder_da.eval()
    model_da.predictor_da.eval()

    all_embeddings = []
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for x, y in loader:
            x_cuda = x.cuda()
            y_cuda = y.cuda()
            emb = model_da.encoder_da(x_cuda)
            pred = model_da.predictor_da(emb)
            score = torch.clamp(1 - torch.mean(torch.abs(pred - y_cuda), dim=1), 0, 1)

            all_embeddings.append(emb.detach().cpu().numpy())
            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
            all_scores.append(score.detach().cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    num_classes = labels_arr.shape[1]
    max_vals = np.max(labels_arr, axis=1)
    min_vals = np.min(labels_arr, axis=1)
    ranges = max_vals - min_vals
    dominant_classes = np.argmax(labels_arr, axis=1)
    camp_ids = np.where(ranges < mixed_margin, num_classes, dominant_classes)

    entropy = _entropy_rows(labels_arr)
    margin = _dominant_margin(labels_arr)
    smoothness = _local_smoothness_proxy(embeddings, labels_arr, k_neighbors=k_neighbors)

    camp_rows = []
    camp_names = list(labels) + ["MIXED"]

    for camp_id in range(num_classes + 1):
        camp_mask = camp_ids == camp_id
        camp_indices = np.nonzero(camp_mask)[0]
        camp_name = camp_names[camp_id]

        if len(camp_indices) == 0:
            camp_rows.append(
                {
                    "camp_id": camp_id,
                    "camp_name": camp_name,
                    "count": 0,
                    "score_mean": np.nan,
                    "score_std": np.nan,
                    "entropy_mean": np.nan,
                    "range_mean": np.nan,
                    "margin_mean": np.nan,
                    "local_label_l1_mean": np.nan,
                    "local_emb_dist_mean": np.nan,
                    "local_variation_ratio_mean": np.nan,
                }
            )
            continue

        camp_rows.append(
            {
                "camp_id": camp_id,
                "camp_name": camp_name,
                "count": int(len(camp_indices)),
                "score_mean": float(np.mean(scores[camp_indices])),
                "score_std": float(np.std(scores[camp_indices])),
                "entropy_mean": float(np.mean(entropy[camp_indices])),
                "range_mean": float(np.mean(ranges[camp_indices])),
                "margin_mean": float(np.mean(margin[camp_indices])),
                "local_label_l1_mean": float(np.mean(smoothness["local_label_l1"][camp_indices])),
                "local_emb_dist_mean": float(np.mean(smoothness["local_emb_dist"][camp_indices])),
                "local_variation_ratio_mean": float(np.mean(smoothness["local_variation_ratio"][camp_indices])),
            }
        )

    summary_df = pd.DataFrame(camp_rows)
    summary_df["frequency_rank"] = summary_df["local_variation_ratio_mean"].rank(
        method="min", ascending=False
    )
    summary_df = summary_df.sort_values(
        by=["local_variation_ratio_mean", "entropy_mean"], ascending=[False, False]
    ).reset_index(drop=True)

    detail_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(scores)),
            "camp_id": camp_ids,
            "camp_name": [camp_names[i] for i in camp_ids],
            "score": scores,
            "entropy": entropy,
            "range": ranges,
            "margin": margin,
            "local_label_l1": smoothness["local_label_l1"],
            "local_emb_dist": smoothness["local_emb_dist"],
            "local_variation_ratio": smoothness["local_variation_ratio"],
        }
    )
    for idx, ctype in enumerate(labels):
        detail_df[f"true_{ctype}"] = labels_arr[:, idx]
        detail_df[f"pred_{ctype}"] = preds[:, idx]

    if out_dir is not None:
        summary_csv = os.path.join(out_dir, "camp_frequency_summary.csv")
        detail_csv = os.path.join(out_dir, "camp_frequency_details.csv")
        figure_path = os.path.join(out_dir, "camp_frequency_proxy.png")
        entropy_path = os.path.join(out_dir, "camp_entropy_proxy.png")

        summary_df.to_csv(summary_csv, index=False)
        detail_df.sort_values(by="local_variation_ratio", ascending=False).to_csv(detail_csv, index=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df = summary_df.sort_values(by="local_variation_ratio_mean", ascending=True)
        ax.bar(plot_df["camp_name"], plot_df["local_variation_ratio_mean"], color="#4C78A8")
        ax.set_ylabel("Local variation ratio")
        ax.set_title("Camp frequency proxy: higher means more high-frequency-like")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(figure_path, dpi=180)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        plot_df2 = summary_df.sort_values(by="entropy_mean", ascending=True)
        ax2.bar(plot_df2["camp_name"], plot_df2["entropy_mean"], color="#F58518")
        ax2.set_ylabel("Normalized label entropy")
        ax2.set_title("Camp label entropy: lower means more low-frequency-like")
        ax2.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(entropy_path, dpi=180)
        plt.close(fig2)
    else:
        summary_csv = ""
        detail_csv = ""
        figure_path = ""
        entropy_path = ""

    lowest_row = summary_df.sort_values(by="local_variation_ratio_mean", ascending=True).iloc[0]
    highest_row = summary_df.sort_values(by="local_variation_ratio_mean", ascending=False).iloc[0]

    return {
        "summary_df": summary_df,
        "detail_df": detail_df,
        "summary_csv": summary_csv,
        "detail_csv": detail_csv,
        "figure_path": figure_path,
        "entropy_path": entropy_path,
        "lowest_camp": lowest_row["camp_name"],
        "highest_camp": highest_row["camp_name"],
        "lowest_proxy": float(lowest_row["local_variation_ratio_mean"]),
        "highest_proxy": float(highest_row["local_variation_ratio_mean"]),
        "camp_names": camp_names,
    }
