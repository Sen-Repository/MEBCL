from imblearn.over_sampling import SMOTE
import torch
from scipy.stats import entropy
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

def calculate_feature_entropy(feature_series, bin_count=10):

    if feature_series.nunique() <= 1:
        return 0.0
    try:
        bin_edges = np.linspace(feature_series.min(), feature_series.max(), bin_count + 1)
        hist, _ = np.histogram(feature_series, bins=bin_edges, density=True)
        valid_hist = hist[hist > 0]
        return entropy(valid_hist, base=2) if len(valid_hist) > 0 else 0.0
    except Exception as e:
        print(f"特征熵计算错误: {str(e)}")
        return 0.0


def perform_feature_selection(omics_data, data_type, selection_config):

    valid_features = omics_data.var(axis=0) > selection_config['variance_threshold']
    filtered_data = omics_data.loc[:, valid_features]

    entropy_scores = filtered_data.apply(
        lambda col: calculate_feature_entropy(col, selection_config['bin_count']),
        axis=0
    )

    top_k = selection_config['feature_counts'].get(data_type, selection_config['default_top_k'])
    top_k = min(top_k, filtered_data.shape[1])
    selected_features = entropy_scores.sort_values(ascending=False).index[:top_k]
    return selected_features

def myNormalize(train_data, test_data):
    mins = np.min(train_data, axis=0)
    maxs = np.max(train_data, axis=0)
    denominator = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    train_norm = (train_data - mins) / denominator
    test_norm = (test_data - mins) / denominator
    return np.clip(train_norm, 0.0, 1.0), np.clip(test_norm, 0.0, 1.0)
def z_score_normalize(train_data, test_data):
    means = np.mean(train_data, axis=0)
    stds = np.std(train_data, axis=0)
    stds = np.where(stds == 0, 1.0, stds)

    train_norm = (train_data - means) / stds
    test_norm = (test_data - means) / stds
    return train_norm, test_norm
class MyDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index]


class AlignedSampler(Sampler):
    def __init__(self, data_source, seed):
        self.num_samples = len(data_source)
        self.rng = np.random.default_rng(seed)
        self.indices = self.rng.permutation(self.num_samples).tolist()
    def __iter__(self):
        yield from self.indices

    def __len__(self):
        return len(self.indices)


def multi_cancer_data_loader(cancer_types, batch_size, seed, view_list, device, fillna_value,selection_config=None):
    if not isinstance(view_list, (list, tuple)):
        raise ValueError("view_list must be a list or tuple")

    if selection_config is None:
        selection_config = {
            'variance_threshold': 0.01,
            'bin_count': 10,
            'feature_counts': {'all': 5000},
            'default_top_k': 5000
        }


    all_cancer_view_data = {}  # {cancer_type: {view: X}}

    for cancer_type in cancer_types:
        view_data = {}
        for view in view_list:
            file_suffix_mapping = {
                'rnaseq': 'preprocessed_RNASeq4',
                'mirna': 'preprocessed_MIRNA4',
                'psi': 'preprocessed_PSI4',
                'cnv': 'preprocessed_CNV4',
                'dnaMeth': 'preprocessed_DNAMeth4',
            }
            file_suffix = file_suffix_mapping[view]
            file_path = f"D:/TCGA-DATA/result/data_pre/data_no_all/TCGA_{cancer_type}/{file_suffix}.csv"

            data = pd.read_csv(file_path, index_col=0)
            data.index = [idx[0:12] for idx in data.index]  # 假设样本ID为索引
            data.columns = [f'col{i}' for i in range(data.shape[1])]

            if data.isnull().values.any():
                print(f"Warning: NaN values found in {file_path}. Filling with {fillna_value}.")
                data = data.fillna(fillna_value)
            view_data[view] = data  # 直接存储原始数据（含样本索引）

        merged_data = pd.concat(view_data.values(), axis=1)
        merged_data.columns = [f'col{i}' for i in range(merged_data.shape[1])]
        all_cancer_view_data[cancer_type] = merged_data

    cancer_data_list = [cancer_values for cancer_values in all_cancer_view_data.values()]
    combined_data = pd.concat(cancer_data_list, axis=0)

    if combined_data.isnull().values.any():
        print(f"Warning: NaN values found in combined data. Filling with {fillna_value}.")
        combined_data = combined_data.fillna(fillna_value)
    combined_data = combined_data.fillna(combined_data.mean())  # 均值填充

    train_val_X, test_X = train_test_split(combined_data, test_size=0.2, random_state=42)


    selected_features = perform_feature_selection(train_val_X, "all", selection_config)

    train_val_X_filtered = train_val_X[selected_features]
    test_X_filtered = test_X[selected_features]

    train_val_X_norm, test_X_norm = z_score_normalize(
        train_val_X_filtered.values,
        test_X_filtered.values
    )

    train_val_X_norm_df = pd.DataFrame(
        train_val_X_norm,
        columns=selected_features,
        index=train_val_X.index
    )
    test_X_norm_df = pd.DataFrame(
        test_X_norm,
        columns=selected_features,
        index=test_X.index
    )
    train_dataset = MyDataset(torch.tensor(train_val_X_norm_df.values, dtype=torch.float32).to(device))
    test_dataset = MyDataset(torch.tensor(test_X_norm_df.values, dtype=torch.float32).to(device))
    full_train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    feature_dim = train_val_X_norm_df.shape[1]

    return full_train_loader, test_loader, feature_dim
