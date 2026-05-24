import os
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import random
from tqdm import tqdm
import time
import numpy as np
from colorama import Fore, Style, init
import pandas as pd
from collections import defaultdict
import warnings
import torch.nn.functional as F
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))
    def forward(self, x):
        out = self.layer(x)
        return out
    
class PrototypeBank(nn.Module):
    def __init__(self, k_size, feature_dim, label_dim):
        super().__init__()
        self.register_buffer('feature', torch.zeros(k_size, feature_dim))
        self.register_buffer('label', torch.zeros(k_size, label_dim))
        self.k_size = k_size

    def update(self, new_features, new_labels):
        self.feature.copy_(new_features)
        self.label.copy_(new_labels)

    def forward(self):
        return self.feature, self.label

def calculate_deconv_score(pred, gt):

    mae = torch.mean(torch.abs(pred - gt), dim=1)
    max_error = torch.max(torch.abs(pred - gt), dim=1)[0]
    
    # 按照 0.7 和 0.3 的权重融合
    # combined_error = (0.7 * mae) + (0.3 * max_error)
    # 考虑cosine
    return torch.clamp((1 - mae), 0, 1)

def RefreshPrototype(model_list, dataloder, bank):
        
        encoder = model_list[0]
        predictor = model_list[1]
        encoder.eval()
        predictor.eval()

        old_labels = bank.label.clone()

        all_x_list = []
        all_y_list = []
        all_score_list = []

        with torch.no_grad():
            for source_x, source_y in dataloder:
                # 数据处理
                src_x = source_x.cuda()
                src_y = source_y.cuda()

                embedding_source = encoder(src_x)
                frac_pred = predictor(embedding_source)
                batch_scores = calculate_deconv_score(frac_pred, src_y)

                all_x_list.append(src_x)
                all_y_list.append(src_y)
                all_score_list.append(batch_scores)

        full_x = torch.cat(all_x_list, dim=0)
        full_y = torch.cat(all_y_list, dim=0)
        full_scores = torch.cat(all_score_list, dim=0)

        _, top_indices = torch.topk(full_scores, k=bank.k_size, largest=True)

        new_features = full_x[top_indices].cuda()
        new_labels = full_y[top_indices].cuda()

        bank.update(new_features, new_labels)

        update_magnitude = torch.mean(torch.abs(new_labels - old_labels)).item()
        avg_bank_score = torch.mean(full_scores[top_indices]).item()

        return update_magnitude, avg_bank_score



def get_cos_sim(feat,bank_feat):
    
        feat_norm = F.normalize(feat, p=2, dim=1)       # [Batch, Dim]
        bank_feat_norm = F.normalize(bank_feat, p=2, dim=1) # [K, Dim]

        sim_matrix = torch.matmul(feat_norm, bank_feat_norm.t())

        return sim_matrix
    
def compute_alignment_loss(feat, label, bank_feat, bank_label):

    s_feat = get_cos_sim(feat, bank_feat)
    s_label = get_cos_sim(label, bank_label)

    loss = F.mse_loss(s_feat, s_label.detach())
    return loss

class demo2(object):
    def __init__(self, num_epochs, batch_size, learning_rate):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.celltype_num = None
        self.labels = None
        self.used_features = None
        self.seed = 2021

        cudnn.deterministic = True
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def demo2_model(self, celltype_num):
        feature_num = len(self.used_features)

        self.encoder_da = nn.Sequential(EncoderBlock(feature_num, 512, 0), 
                                        EncoderBlock(512, 256, 0.3))

        self.predictor_da = nn.Sequential(EncoderBlock(256, 128, 0.2), 
                                          nn.Linear(128, celltype_num), 
                                          nn.Softmax(dim=1))
        self.bank = PrototypeBank(self.batch_size, feature_num, celltype_num).cuda()

        model_da = nn.ModuleList([])
        model_da.append(self.encoder_da)
        model_da.append(self.predictor_da)
        return model_da

    def prepare_dataloader(self, source_data, target_data, valid_data, batch_size):
        ### Prepare data loader for training ###
        # Source dataset
        source_ratios = [source_data.obs[ctype] for ctype in source_data.uns['cell_types']]
        self.source_data_x = source_data.X.astype(np.float32)
        self.source_data_y = np.array(source_ratios, dtype=np.float32).transpose()
        
        tr_data = torch.FloatTensor(self.source_data_x)
        tr_labels = torch.FloatTensor(self.source_data_y)
        source_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)

        # Extract celltype and feature info
        self.labels = source_data.uns['cell_types']
        self.celltype_num = len(self.labels)
        self.used_features = list(source_data.var_names)

        # Target dataset
        self.target_data_x = target_data.X.astype(np.float32)
        target_ratios = [target_data.obs[ctype] for ctype in self.labels]
        self.target_data_y = np.array(target_ratios, dtype=np.float32).transpose()
        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        target_dataset = Data.TensorDataset(te_data, te_labels)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False)
        
        # valid dataset
        self.valid_data_x = valid_data.X.astype(np.float32)
        valid_ratios = [valid_data.obs[ctype] for ctype in self.labels]
        self.valid_data_y = np.array(valid_ratios, dtype=np.float32).transpose()
        va_data = torch.FloatTensor(self.valid_data_x)
        va_labels = torch.FloatTensor(self.valid_data_y)
        valid_dataset = Data.TensorDataset(va_data, va_labels)
        self.valid_target_loader = Data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
        


    def train(self, source_data, target_data, valid_data, patience):
        ### prepare model structure ###
        self.prepare_dataloader(source_data, target_data, valid_data, self.batch_size)
        self.model_da = self.demo2_model(self.celltype_num).cuda()

        # 设置优化器
        optimizer = torch.optim.Adam([
            {'params': self.encoder_da.parameters()},
            {'params': self.predictor_da.parameters()}
        ], lr=self.learning_rate)
        criterion_pred = nn.L1Loss().cuda()

        # 初始化变量
        counter = 0
        best_model_weights = None
        best_rmse = float('inf')
        pred_loss_list = []
        valid_rmse_list = []

        # 颜色定义
        HEADER = Fore.CYAN
        METRIC = Fore.GREEN
        WARNING = Fore.YELLOW
        BEST = Fore.MAGENTA
        RESET = Style.RESET_ALL

        print(f"\n{HEADER}===== Starting Training (Total Epochs: {self.num_epochs}) =====")
        print(f"Patience for early stopping: {patience} epochs")
        print(f"Batch size: {self.batch_size}, Learning rate: {self.learning_rate}{RESET}\n")

        for epoch in range(self.num_epochs):

            if epoch > 5:
                up_mag, bank_score = RefreshPrototype(self.model_da, self.train_source_loader, self.bank)
                print(f" 🔍 {Fore.YELLOW}[Bank Monitor]{Style.RESET_ALL} "
                    f"Update Mag: {Fore.WHITE}{up_mag:.6f}{Style.RESET_ALL} | "
                    f"Avg Score: {Fore.GREEN}{bank_score:.4f}{Style.RESET_ALL}")

            self.model_da.train()
            total_iterations = len(self.train_source_loader)

            # 创建进度条
            pbar = tqdm(total=total_iterations, 
                        desc=f"Epoch {epoch+1}/{self.num_epochs}", 
                        dynamic_ncols=True,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} batches')

            pred_loss_epoch = 0.

            for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
                # 数据处理
                src_x = source_x.cuda()
                src_y = source_y.cuda()

                # 前向传播和优化
                embedding_source = self.encoder_da(src_x)
                frac_pred = self.predictor_da(embedding_source)

                # 计算损失
                pred_loss = criterion_pred(frac_pred, src_y)

                bank_feature = self.bank.feature
                with torch.no_grad():
                    bank_feat = self.encoder_da(bank_feature)
                    bank_pred = self.predictor_da(bank_feat)
                    bank_label = self.bank.label

                loss_align = compute_alignment_loss(embedding_source, frac_pred, bank_feat, bank_pred)
                
                if epoch < 5:
                    alpha = 0.0
                else:
                    alpha = 0.1 

                loss_total = pred_loss + alpha * loss_align

                # print(f"pred_loss:{pred_loss}   loss_align:{loss_align}   loss_total:{loss_total}")

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                # 收集损失
                pred_loss_epoch += pred_loss.item()

                # 更新进度条
                pbar.update(1)
                if batch_idx % 10 == 0:  # 每10个batch更新一次
                    avg_pred = pred_loss_epoch / (batch_idx + 1)

                    pbar.set_postfix({
                        'pred': f'{avg_pred:.4f}'
                    })

            # 关闭进度条
            pbar.close()

            




            # 计算本轮平均损失
            pred_avg = pred_loss_epoch / total_iterations

            # 保存本轮损失
            pred_loss_list.append(pred_avg)

            # 验证模型性能
            valid_rmse = self.evaluate(self.valid_target_loader)
            valid_rmse_list.append(valid_rmse)

            # 输出本轮结果
            print(f"{HEADER}[Ep {epoch+1}] | "
                 f"Pred: {METRIC}{pred_avg:.4f}{RESET} | "
                 f"Valid RMSE: {METRIC}{valid_rmse:.4f}{RESET}")

            # 检查是否需要保存模型
            if valid_rmse < best_rmse:
                best_rmse = valid_rmse
                counter = 0
                best_model_weights = {
                    'encoder': copy.deepcopy(self.encoder_da.state_dict()),
                    'predictor': copy.deepcopy(self.predictor_da.state_dict())
                }
                print(f"  {BEST}★ New best RMSE! Model saved.{RESET}")
            else:
                counter += 1
                print(f"  {WARNING}↯ No improvement ({counter}/{patience}){RESET}")

            # 检查是否需要早停
            if counter >= patience:
                print(f"{HEADER}\nEarly stopping triggered at epoch {epoch+1}!")
                print(f"Best RMSE achieved: {best_rmse:.4f}{RESET}\n")
                break

        # 最终训练报告
        print(f"\n{HEADER}===== Training Complete! =====")
        print(f"Total epochs: {len(pred_loss_list)}/{self.num_epochs}")
        print(f"Best RMSE: {BEST}{best_rmse:.4f}{RESET}")
        print(f"Final loss: Pred={pred_loss_list[-1]:.4f}")
        print("===============================")
        print(f"{RESET}")

        return pred_loss_list, best_model_weights

    def visualize_final_prototypes(self, source_data, out_dir="demo2_prototype_vis"):
        os.makedirs(out_dir, exist_ok=True)

        if self.labels is None:
            self.labels = source_data.uns['cell_types']

        source_x = source_data.X.astype(np.float32)
        source_ratios = [source_data.obs[ctype] for ctype in self.labels]
        source_y = np.array(source_ratios, dtype=np.float32).transpose()

        src_data = torch.FloatTensor(source_x)
        src_labels = torch.FloatTensor(source_y)
        src_dataset = Data.TensorDataset(src_data, src_labels)
        src_loader = Data.DataLoader(dataset=src_dataset, batch_size=self.batch_size, shuffle=False)

        self.model_da.eval()
        all_embeddings = []
        all_preds = []
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for x, y in src_loader:
                x_cuda = x.cuda()
                y_cuda = y.cuda()
                emb = self.encoder_da(x_cuda)
                pred = self.predictor_da(emb)
                score = calculate_deconv_score(pred, y_cuda)

                all_embeddings.append(emb.detach().cpu().numpy())
                all_preds.append(pred.detach().cpu().numpy())
                all_labels.append(y.detach().cpu().numpy())
                all_scores.append(score.detach().cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        preds = np.concatenate(all_preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        scores = np.concatenate(all_scores, axis=0)

        k_size = min(self.bank.k_size, len(scores)) if hasattr(self, 'bank') else min(self.batch_size, len(scores))
        top_indices = np.argpartition(scores, -k_size)[-k_size:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        is_prototype = np.zeros(len(scores), dtype=bool)
        is_prototype[top_indices] = True

        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        pca_2d = centered @ vt[:2].T
        var_ratio = (s ** 2) / np.sum(s ** 2)
        pc1_ratio = float(var_ratio[0]) if len(var_ratio) > 0 else 0.0
        pc2_ratio = float(var_ratio[1]) if len(var_ratio) > 1 else 0.0

        fig_scatter = os.path.join(out_dir, "prototype_feature_scatter.png")
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_2d[:, 0], pca_2d[:, 1], s=10, c='lightgray', alpha=0.35, label='all source samples')
        sc = plt.scatter(
            pca_2d[is_prototype, 0],
            pca_2d[is_prototype, 1],
            s=30,
            c=scores[is_prototype],
            cmap='viridis',
            edgecolors='black',
            linewidths=0.3,
            label=f'prototype top-{k_size}',
        )
        plt.colorbar(sc, label='deconv score')
        plt.xlabel(f'PC1 ({pc1_ratio:.2%})')
        plt.ylabel(f'PC2 ({pc2_ratio:.2%})')
        plt.title('Prototype distribution in source feature space (PCA)')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(fig_scatter, dpi=180)
        plt.close()

        fig_hist = os.path.join(out_dir, "prototype_score_distribution.png")
        plt.figure(figsize=(8, 5))
        plt.hist(scores, bins=40, alpha=0.5, label='all source scores')
        plt.hist(scores[is_prototype], bins=min(20, k_size), alpha=0.8, label='prototype scores')
        plt.xlabel('deconv score')
        plt.ylabel('count')
        plt.title('Score distribution: all samples vs prototypes')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(fig_hist, dpi=180)
        plt.close()

        true_dom_idx = np.argmax(labels, axis=1)
        pred_dom_idx = np.argmax(preds, axis=1)

        detail_df = pd.DataFrame({
            'sample_index': np.arange(len(scores)),
            'score': scores,
            'is_prototype': is_prototype,
            'true_dominant_type': [self.labels[i] for i in true_dom_idx],
            'pred_dominant_type': [self.labels[i] for i in pred_dom_idx],
        })

        for i, ctype in enumerate(self.labels):
            detail_df[f'true_{ctype}'] = labels[:, i]
            detail_df[f'pred_{ctype}'] = preds[:, i]

        detail_df = detail_df.sort_values(by='score', ascending=False).reset_index(drop=True)
        detail_csv = os.path.join(out_dir, 'prototype_sample_details.csv')
        top_csv = os.path.join(out_dir, 'prototype_topk_samples.csv')
        detail_df.to_csv(detail_csv, index=False)
        detail_df[detail_df['is_prototype']].to_csv(top_csv, index=False)

        return {
            'scatter_path': fig_scatter,
            'score_hist_path': fig_hist,
            'detail_csv': detail_csv,
            'topk_csv': top_csv,
            'k_size': int(k_size),
            'topk_score_mean': float(np.mean(scores[is_prototype])),
            'all_score_mean': float(np.mean(scores)),
        }

    
    def prediction(self, data_test):
        self.model_da.eval()
        preds, gt = None, None
        for batch_idx, (x, y) in enumerate(data_test):
            logits = self.predictor_da(self.encoder_da(x.cuda())).detach().cpu().numpy()
            frac = y.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
            gt = frac if gt is None else np.concatenate((gt, frac), axis=0)

        target_preds = pd.DataFrame(preds, columns=self.labels)
        ground_truth = pd.DataFrame(gt, columns=self.labels)
        return target_preds, ground_truth
    
    def evaluate(self, valid_data):
        final_preds_target, ground_truth_target = self.prediction(valid_data)
        _ = []
        for label in self.labels:  
            rmse = np.sqrt(np.mean((final_preds_target[label] - ground_truth_target[label]) ** 2))  
            _.append(rmse)  

            # 计算平均RMSE  
        avg_rmse = np.mean(_)  
        return avg_rmse