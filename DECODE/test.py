import os
# Set your working directory to the DECODE directory
os.chdir('E:\Desktop\DECODE-main')

import numpy as np
import pickle
import anndata as ad
from sklearn.model_selection import train_test_split
import warnings
import copy

from data.data_process import data_process
from model.deconv_model_with_stage_2 import MBdeconv
from model.utils import *
from model.stage2 import *

seed = 2021
torch.manual_seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")

# Define the cell types of interest and read the corresponding single-cell matrix data.
type_list = ['Luminal_Macrophages', 'Type 2 alveolar', 'Fibroblasts', 'Dendritic cells']
noise = ['Neutrophils']
train_data_file = 'data/lung_rna/296C_train.h5ad'
test_data_file = 'data/lung_rna/302C_test.h5ad'
train_data = ad.read_h5ad(train_data_file)
test_data = ad.read_h5ad(test_data_file)

# Select the corresponding cells based on the cell types of interest.
if noise:
    data_h5ad_noise = test_data[test_data.obs['CellType'].isin(noise)]
    data_h5ad_noise.obs.reset_index(drop=True, inplace=True)
# extract selected cells
train_data = train_data[train_data.obs['CellType'].isin(type_list)]
train_data.obs.reset_index(drop=True, inplace=True)
test_data = test_data[test_data.obs['CellType'].isin(type_list)]
test_data.obs.reset_index(drop=True, inplace=True)
print('selected cells:', train_data)
print('noise cells:', data_h5ad_noise)

# Define the key parameters in the simulated experiment,
# including the number of training and testing data entries and
# the capacity of pseudo-organized cells. The number of artificial noise cells
# used in stage three of the mixing phase is typically set to be the same as that of the pseudotissue cells.

dp = data_process(type_list, train_sample_num=6000, tissue_name='lung_rna',
                  test_sample_num=1000, sample_size=30, num_artificial_cells=30)

# data_h5ad_noise is a dataset used to add unknown cell types to the test dataset
dp.fit(train_data, test_data, data_h5ad_noise)

# Read the dataset, where train is used for training, test is a mixed test set from different donors,
# and test_with_noise contains unseen cells from train mixed in different proportions,
# with the same labels as the test set

with open(f'data/lung_rna/lung_rna{len(type_list)}cell.pkl', 'rb') as f:
    train = pickle.load(f)
    test = pickle.load(f)
    test_with_noise = pickle.load(f)

train_x_sim, train_with_noise_1, train_with_noise_2, train_y = train
test_x_sim, test_y = test

# Partition a portion of the test dataset for evaluating performance to serve the early stopping mechanism.
valid_size = 1000

valid_x_sim = train_x_sim[:valid_size]
valid_with_noise_1 = train_with_noise_1[:valid_size]
valid_with_noise_2 = train_with_noise_2[:valid_size]
valid_y = train_y[:valid_size]

train_x_sim = train_x_sim[valid_size:]
train_with_noise_1 = train_with_noise_1[valid_size:]
train_with_noise_2 = train_with_noise_2[valid_size:]
train_y = train_y[valid_size:]

test_dataset = TestCustomDataset(test_x_sim, test_y)
valid_dataset = TestCustomDataset(valid_x_sim, valid_y)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

train_dataset = TrainCustomDataset(train_x_sim, train_with_noise_1, train_with_noise_2, train_y)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)


source_data = data2h5ad(train_x_sim, train_y, type_list)
target_data = data2h5ad(test_x_sim, test_y, type_list)
valid_data = data2h5ad(valid_x_sim, valid_y, type_list)

num_feat = 3346
feat_map_w = 256
feat_map_h = 10
num_cell_type = len(type_list)
patience = 10
epoches = 200
Alpha = 1
Beta = 1
model_save_name = 'lung_rna'

# Train stage 2, returning the training loss and the best encoder parameters.
model_da = DANN(epoches, 50, 0.0001)
pred_loss, disc_loss, disc_loss_DA, best_model_weights = model_da.train(source_data, target_data, valid_data, patience = 3)

model = MBdeconv(num_feat, feat_map_w, feat_map_h, num_cell_type, epoches, Alpha, Beta, train_dataloader, valid_dataloader)

# Train stage 3, reading the parameters of stage 2 encoder before training.
device = torch.device('cuda')
if model.gpu_available:
    model = model.to(model.gpu)
model_da.encoder_da.load_state_dict(best_model_weights['encoder'])
encoder_params = copy.deepcopy(model_da.encoder_da.state_dict())
model.encoder.load_state_dict(encoder_params)
loss1_list, loss2_list, nce_loss_list = model.train_model(model_save_name, True, patience)

model_test = MBdeconv(num_feat, feat_map_w, feat_map_h, num_cell_type, epoches, Alpha, Beta, train_dataloader, valid_dataloader)

# Perform inference on the test dataset in Stage 4 and obtain the overall CCC, RMSE, and Correlation values.
model_test.load_state_dict(torch.load('save_models/3346/lung_rna.pt'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_test.to(device)
model_test.eval()
CCC, RMSE, Corr, pred, gt = predict(test_dataloader, type_list, model_test, True)

CCC, RMSE, Corr
print(f"  CCC  (Concordance Correlation Coefficient): {CCC:.4f}")
print(f"  RMSE (Root Mean Square Error)             : {RMSE:.4f}")
print(f"  Corr (Pearson Correlation)                : {Corr:.4f}")