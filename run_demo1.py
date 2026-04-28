import os
import pickle
import random
import warnings
from datetime import datetime

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from tqdm import tqdm


from demo1 import demo1 as DemoModel

warnings.filterwarnings("ignore")


# =====================
# Config
# =====================
WORKDIR = r"E:/Desktop/new"
TYPE_LIST = ["Luminal_Macrophages", "Type 2 alveolar", "Fibroblasts", "Dendritic cells"]

TRAIN_COUNTS_FILE = "296C_counts.txt"
TRAIN_CELLTYPES_FILE = "296C_celltypes.txt"
TEST_BULK_FILE = "302C_bulk_X.txt"
TEST_BULK_OBS_FILE = "302C_bulk_obs.txt"
TISSUE_NAME = "lung_rna"
RANDOM_TYPE = "CellType"

TRAIN_SAMPLE_NUM = 6000
TEST_SAMPLE_NUM = 1000
SAMPLE_SIZE = 30
VALID_SIZE = 1000

EPOCHS = 200
BATCH_SIZE = 50
LEARNING_RATE = 0.0001
PATIENCE = 3

OUT_PRED = "demo1_preds.csv"
OUT_METRICS = "demo1_metrics_summary.csv"
OUT_TYPEWISE = "demo1_typewise_ccc.csv"

USE_CACHE = False


class data_process(object):
    def __init__(
        self,
        type_list,
        tissue_name,
        sample_size=45,
        train_sample_num=6000,
        test_sample_num=1000,
        random_type="CellType",
    ):
        self.tissue_name = tissue_name
        self.random_type = random_type
        self.type_list = type_list
        self.train_sample_num = train_sample_num
        self.celltype_num = len(self.type_list)
        self.sample_size = sample_size
        self.test_sample_num = test_sample_num

    def build_pseudo_bulk(self, data, purpose):
        data_x = data.X
        if sparse.issparse(data_x):
            data_x = data_x.toarray()
        data_x = pd.DataFrame(
            np.asarray(data_x, dtype=np.float32),
            columns=data.var_names.astype(str),
        ).fillna(0)
        data_x = data_x.clip(lower=0)
        data_y = pd.DataFrame(data.obs[self.random_type]).reset_index(drop=True)

        x_sim = []
        y = []

        if purpose == "train":
            total_num = self.train_sample_num
        else:
            total_num = self.test_sample_num

        print(f"Generating {purpose} pseudo_bulk samples...")
        with tqdm(total=total_num, desc=f"{purpose} Samples") as pbar:
            while len(x_sim) < total_num:
                result = self.mix_cells(data_x, data_y, cell_type_list=self.type_list)
                if result is None:
                    continue
                sample, label = result
                x_sim.append(sample)
                y.append(label)
                pbar.update(1)
        return x_sim, y

    def mix_cells(self, x, y, cell_type_list):
        fracs = self.mixup_fraction(len(cell_type_list))
        samp_fracs = np.multiply(fracs, self.sample_size)
        samp_fracs = list(map(round, samp_fracs))

        if sum(samp_fracs) == 0:
            return None

        fracs = np.divide(samp_fracs, sum(samp_fracs))
        fracs_complete = [0] * len(cell_type_list)
        for i, act in enumerate(cell_type_list):
            idx = cell_type_list.index(act)
            fracs_complete[idx] = fracs[i]

        artificial_samples = []
        for i, ct in enumerate(cell_type_list):
            cells_sub = x.loc[y[self.random_type] == ct]
            if cells_sub.shape[0] > 0 and samp_fracs[i] <= len(cells_sub):
                cells_fraction = np.random.randint(0, cells_sub.shape[0], samp_fracs[i])
                cells_sub = cells_sub.iloc[cells_fraction, :]
                artificial_samples.append(cells_sub)
            else:
                return None

        df_samp = pd.concat(artificial_samples, axis=0).sum(axis=0)
        return df_samp, fracs_complete

    def mixup_fraction(self, cell_num):
        fracs = np.random.rand(cell_num)
        fracs_sum = np.sum(fracs)
        if fracs_sum == 0:
            return np.repeat(1.0 / cell_num, cell_num)
        fracs = np.divide(fracs, fracs_sum)
        return fracs

    def normalize(self, series_list):
        normalized_series_list = []
        for series in series_list:
            max_value = series.max()
            if max_value == 0:
                normalized_series = series
            else:
                normalized_series = series / max_value
            normalized_series_list.append(normalized_series)
        return normalized_series_list

    def fit(self, train_data, test_data):
        os.makedirs(os.path.join("data", self.tissue_name), exist_ok=True)
        path = os.path.join("data", self.tissue_name, f"{self.tissue_name}{len(self.type_list)}cell.pkl")

        if USE_CACHE and os.path.exists(path):
            print("The data processing is complete")
            return path

        train_x_sim, train_y = self.build_pseudo_bulk(train_data, "train")

        train_x_sim = self.normalize(train_x_sim)
        train = [train_x_sim, train_y]
        test_placeholder = []

        with open(path, "wb") as f:
            pickle.dump(train, f)
            pickle.dump(test_placeholder, f)

        train_data.write_h5ad(os.path.join("data", self.tissue_name, "ref_cell.h5ad"))
        print("The data processing is complete")
        return path


def read_expression_matrix(path):
    # Input format: rows are genes, columns are samples/cells.
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def load_train_adata_from_296c(counts_file, celltypes_file, keep_types):
    counts_df = read_expression_matrix(counts_file)
    celltypes_df = pd.read_csv(celltypes_file, sep="\t")

    if RANDOM_TYPE not in celltypes_df.columns:
        raise ValueError(f"{celltypes_file} missing column: {RANDOM_TYPE}")

    n_cells = counts_df.shape[1]
    n_cell_labels = len(celltypes_df)

    # Support both matrix layouts:
    # 1) cells x genes -> rows match cell labels
    # 2) genes x cells -> cols match cell labels
    if counts_df.shape[0] == n_cell_labels:
        X = counts_df.to_numpy(dtype=np.float32)
        var_names = counts_df.columns.astype(str)
    elif counts_df.shape[1] == n_cell_labels:
        X = counts_df.T.to_numpy(dtype=np.float32)
        var_names = counts_df.index.astype(str)
    else:
        raise ValueError(
            f"Cannot align {counts_file} with {celltypes_file}: "
            f"counts shape={counts_df.shape}, celltype rows={n_cell_labels}"
        )

    adata = ad.AnnData(X)
    adata.var_names = var_names
    adata.obs[RANDOM_TYPE] = celltypes_df[RANDOM_TYPE].astype(str).to_numpy()

    adata = adata[adata.obs[RANDOM_TYPE].isin(keep_types)]
    adata.obs.reset_index(drop=True, inplace=True)
    return adata


def load_test_bulk_from_302c(bulk_file, gene_order, cell_types):
    bulk_df = read_expression_matrix(bulk_file)
    gene_order = pd.Index(np.asarray(gene_order, dtype=str))
    bulk_genes = pd.Index(bulk_df.index.astype(str))

    if bulk_df.shape[0] != len(gene_order):
        raise ValueError(
            f"Gene count mismatch: bulk genes={bulk_df.shape[0]}, train genes={len(gene_order)}"
        )

    if not bulk_genes.equals(gene_order):
        missing_in_bulk = gene_order.difference(bulk_genes)
        extra_in_bulk = bulk_genes.difference(gene_order)
        if len(missing_in_bulk) > 0 or len(extra_in_bulk) > 0:
            raise ValueError(
                "Gene name mismatch between train and bulk. "
                f"missing_in_bulk={len(missing_in_bulk)}, extra_in_bulk={len(extra_in_bulk)}"
            )
        raise ValueError(
            "Gene order mismatch between train and bulk. "
            "Please reorder bulk genes to exactly match training gene order."
        )

    X = bulk_df.T.to_numpy(dtype=np.float32)  # samples x genes
    # Match pseudo-bulk normalization: each sample divided by its own max value.
    row_max = np.max(X, axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    X = X / row_max

    target_adata = ad.AnnData(X)
    target_adata.var_names = bulk_genes

    # Placeholder labels: required by demo1.py dataloader shape, not used for training.
    for ctype in cell_types:
        target_adata.obs[ctype] = np.zeros(X.shape[0], dtype=np.float32)
    target_adata.uns["cell_types"] = list(cell_types)
    return target_adata


def data2h5ad(x_list, y_list, type_list, feature_names=None):
    X = np.asarray([np.asarray(x) for x in x_list], dtype=np.float32)
    Y = np.asarray(y_list, dtype=np.float32)

    adata = ad.AnnData(X)
    if feature_names is None:
        feature_names = [f"gene_{i}" for i in range(X.shape[1])]
    adata.var_names = pd.Index(np.asarray(feature_names, dtype=str))
    for i, ctype in enumerate(type_list):
        adata.obs[ctype] = Y[:, i]
    adata.uns["cell_types"] = list(type_list)
    return adata


def load_bulk_obs(path, cell_types):
    df = pd.read_csv(path, sep="\t", index_col=0)
    missing = [c for c in cell_types if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[cell_types].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def ccc(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    corr = np.corrcoef(gt, pred)[0][1]
    if np.isnan(corr):
        corr = 0.0
    numerator = 2 * corr * np.std(gt) * np.std(pred)
    denominator = np.var(gt) + np.var(pred) + (np.mean(gt) - np.mean(pred)) ** 2
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_metrics(pred_df, gt_df, cell_types):
    pred = pred_df[cell_types].copy()
    gt = gt_df[cell_types].copy()
    x = pred.to_numpy().reshape(-1)
    y = gt.to_numpy().reshape(-1)

    pearson = float(np.corrcoef(x, y)[0][1])
    if np.isnan(pearson):
        pearson = 0.0
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    ccc_score = ccc(x, y)

    return pd.DataFrame([
        {
            "Pearson": pearson,
            "CCC": ccc_score,
            "RMSE": rmse,
            "timestamp": datetime.now().isoformat(),
        }
    ])


def compute_typewise_ccc(pred_df, gt_df, cell_types):
    rows = []
    for ct in cell_types:
        rows.append({"type": ct, "CCC": ccc(pred_df[ct].to_numpy(), gt_df[ct].to_numpy())})
    return pd.DataFrame(rows)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("demo1.py uses cuda directly. Please run on a CUDA-enabled machine.")

    os.chdir(WORKDIR)

    seed = 2021
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_data = load_train_adata_from_296c(TRAIN_COUNTS_FILE, TRAIN_CELLTYPES_FILE, TYPE_LIST)
    print("selected train cells:", train_data)

    dp = data_process(
        TYPE_LIST,
        train_sample_num=TRAIN_SAMPLE_NUM,
        tissue_name=TISSUE_NAME,
        test_sample_num=TEST_SAMPLE_NUM,
        sample_size=SAMPLE_SIZE,
        random_type=RANDOM_TYPE,
    )
    # Here test_data is only used to satisfy fit signature; pseudo test will be ignored later.
    pkl_path = dp.fit(train_data, train_data)

    with open(pkl_path, "rb") as f:
        train = pickle.load(f)
        _ = pickle.load(f)

    train_x_sim, train_y = train

    valid_x_sim = train_x_sim[:VALID_SIZE]
    valid_y = train_y[:VALID_SIZE]
    train_x_sim = train_x_sim[VALID_SIZE:]
    train_y = train_y[VALID_SIZE:]

    source_data = data2h5ad(train_x_sim, train_y, TYPE_LIST, feature_names=train_data.var_names)
    valid_data = data2h5ad(valid_x_sim, valid_y, TYPE_LIST, feature_names=train_data.var_names)
    target_data = load_test_bulk_from_302c(TEST_BULK_FILE, source_data.var_names, TYPE_LIST)

    model_da = DemoModel(EPOCHS, BATCH_SIZE, LEARNING_RATE)
    pred_loss_list, best_model_weights = model_da.train(source_data, target_data, valid_data, patience=PATIENCE)

    model_da.encoder_da.load_state_dict(best_model_weights["encoder"])
    model_da.predictor_da.load_state_dict(best_model_weights["predictor"])

    pred_df, _ = model_da.prediction(model_da.test_target_loader)
    pred_df.to_csv(OUT_PRED, index=False)

    true_obs_df = load_bulk_obs(TEST_BULK_OBS_FILE, TYPE_LIST)

    # Align by index to avoid any sample-order mismatch.
    pred_eval = pred_df.copy()
    pred_eval.index = pred_eval.index.astype(str)
    true_eval = true_obs_df.copy()
    true_eval.index = true_eval.index.astype(str)
    common_idx = pred_eval.index.intersection(true_eval.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping sample indices between predictions and bulk obs.")

    pred_eval = pred_eval.loc[common_idx, TYPE_LIST]
    true_eval = true_eval.loc[common_idx, TYPE_LIST]

    metrics_df = compute_metrics(pred_eval, true_eval, TYPE_LIST)
    typewise_df = compute_typewise_ccc(pred_eval, true_eval, TYPE_LIST)
    metrics_df.to_csv(OUT_METRICS, index=False)
    typewise_df.to_csv(OUT_TYPEWISE, index=False)

    print(f"Saved: {OUT_PRED}")
    print(f"Saved: {OUT_METRICS}")
    print(f"Saved: {OUT_TYPEWISE}")
    print(f"Final train loss: {pred_loss_list[-1]:.6f}")
    print(
        "Eval summary | "
        f"Pearson: {metrics_df.loc[0, 'Pearson']:.6f}, "
        f"CCC: {metrics_df.loc[0, 'CCC']:.6f}, "
        f"RMSE: {metrics_df.loc[0, 'RMSE']:.6f}"
    )


if __name__ == "__main__":
    main()