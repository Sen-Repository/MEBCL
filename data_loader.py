import pandas as pd
import numpy as np
import torch
from imblearn.under_sampling import EditedNearestNeighbours
from torch.utils.data import Dataset, DataLoader, Sampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import make_pipeline
from sklearn.base import clone
from scipy.stats import entropy
import os

class StratifiedBatchSampler(Sampler):

    def __init__(self, labels, batch_size, shuffle=True):
        self.labels = np.array(labels).astype(np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 计算每个类别的样本索引和全局分布
        self.class_indices = {cls: np.where(self.labels == cls)[0]
                              for cls in np.unique(self.labels)}
        self.class_dist = np.bincount(self.labels) / len(self.labels)

    def __iter__(self):
        indices = []
        for cls, idx in self.class_indices.items():
            # 按全局分布计算每类应取的样本数
            n_samples = max(1, int(self.batch_size * self.class_dist[cls]))
            idx = idx.copy()
            if self.shuffle:
                np.random.shuffle(idx)

            # 循环填充批次
            ptr = 0
            while ptr + n_samples <= len(idx):
                indices.extend(idx[ptr:ptr + n_samples])
                ptr += n_samples
            if ptr < len(idx):
                indices.extend(idx[ptr:])

        # 打乱批次顺序
        if self.shuffle:
            np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return (len(self.labels) + self.batch_size - 1) // self.batch_size


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
    print(f"selected_features shape: {len(selected_features)}")
    return selected_features

def z_score_normalize(train_data, test_data):
    means = np.mean(train_data, axis=0)
    stds = np.std(train_data, axis=0)
    stds = np.where(stds == 0, 1.0, stds)

    train_norm = (train_data - means) / stds
    test_norm = (test_data - means) / stds
    return train_norm, test_norm

class MyDataset(Dataset):

    def __init__(self, data, target, smooth_labels=False, num_classes=None, smoothing=0.1):
        self.data = data
        self.target = target
        self.smooth_labels = smooth_labels

        if smooth_labels:
            assert num_classes is not None,
            self.smooth_targets = (1.0 - smoothing) * torch.zeros(
                (len(target), num_classes),
            ).scatter_(1, target.unsqueeze(1), 1.0) + smoothing / num_classes

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return (self.data[index],
                self.smooth_targets[index] if self.smooth_labels else self.target[index])


class AlignedSampler(Sampler):

    def __init__(self, data_length, seed):
        self.indices = np.random.default_rng(seed).permutation(data_length).tolist()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
def data_loader(cancer_type, batch_size, seed, view_list, device):
    # 配置参数
    stage_path = "test"
    selection_config = {
        'variance_threshold': 0.01,
        'bin_count': 10,
        'feature_counts': {'all': 5000},
        'default_top_k': 5000
    }
    stage_data = pd.read_csv(f"{stage_path}/updated_stage_data.csv", index_col=0)
    stage_data = stage_data[stage_data['ajcc_stage'] != 2.0]

    view_data_cache = {}
    for view in view_list:
        file_map = {
            'rnaseq': 'preprocessed_RNAseq4_gene_name',
            'mirna': 'preprocessed_miRNA4_filtered',
            'psi': 'preprocessed_PSI4_filtered',
            'cnv': 'preprocessed_CNV4_gene_name',
            'dnaMeth': 'preprocessed_DNAMeth4_filtered'
        }

        data = pd.read_csv(f"{stage_path}/TCGA_{cancer_type}/{file_map[view]}.csv", index_col=0)
        data.index = [idx[:15] for idx in data.index]
        data = data.reindex(stage_data.index).fillna(0)  #
        data.columns = [f"{view}_{col}" for col in data.columns]
        view_data_cache[view] = data
    features = pd.concat([view_data_cache[view] for view in view_list], axis=1)
    labels = stage_data['ajcc_stage'].values

    print(f"合并后原始特征数: {features.shape[1]}")
    print(f"唯一特征名数量: {len(features.columns.unique())}")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    all_train_loaders = []
    all_test_loaders = []
    all_ds = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        X_train_raw = features.iloc[train_idx]
        y_train = labels[train_idx]
        X_test_raw = features.iloc[test_idx]
        y_test = labels[test_idx]

        print(f"\nFold {fold + 1} - 开始特征筛选...")
        selected_features = perform_feature_selection(X_train_raw, f"fold_{fold + 1}", selection_config)

        X_train_filtered = X_train_raw[selected_features].values
        X_test_filtered = X_test_raw[selected_features].values

        train_dist = np.bincount(y_train.astype(int))
        test_dist = np.bincount(y_test.astype(int))
        print(f"Fold {fold + 1} - 训练集类别分布: {train_dist}, 测试集: {test_dist}")
        print(f"Fold {fold + 1} - 特征筛选后数量: {X_train_filtered.shape[1]}")

        fill_values = np.mean(X_train_filtered, axis=0)
        X_train_filtered = np.where(np.isnan(X_train_filtered), fill_values, X_train_filtered)
        X_test_filtered = np.where(np.isnan(X_test_filtered), fill_values, X_test_filtered)

        X_train_norm, X_test_norm = z_score_normalize(X_train_filtered, X_test_filtered)

        pipeline = make_pipeline(
            SMOTE(random_state=seed , k_neighbors=5),  # 过采样
            EditedNearestNeighbours(n_neighbors=5)  # 欠采样
        )
        X_train_res, y_train_res = pipeline.fit_resample(X_train_norm, y_train)

        def create_loader(X, y, is_train=True):
            data_tensor = torch.tensor(X, dtype=torch.float32)
            label_tensor = torch.tensor(y, dtype=torch.long)
            dataset = MyDataset(data_tensor, label_tensor, smooth_labels=False)

            if is_train:
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=4,
                    pin_memory=True
                )
            else:
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,  # 顺序加载，保证样本不重复、分布原始
                    drop_last=True,
                    num_workers=4,
                    pin_memory=True
                )
        # 生成加载器并记录
        train_loader = create_loader(X_train_res, y_train_res, is_train=True)
        test_loader = create_loader(X_test_norm, y_test, is_train=False)

        all_train_loaders.append(train_loader)
        all_test_loaders.append(test_loader)
        all_ds.append(X_train_res.shape[1])

    return (all_train_loaders, all_test_loaders, all_ds)