import torch
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, auc, f1_score,confusion_matrix
from data_loader import data_loader
from utils import yaml_config_hook
from Modules.model import MVEBClassifier
from pretrain_vae.vae_models import MVEB

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, reduction='mean'):


        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def load_pretrained_model(model, pretrained_path, device):
    """加载预训练模型（支持包含额外信息的检查点）"""
    checkpoint = torch.load(pretrained_path, map_location=device)

    if 'net' in checkpoint:
        state_dict = checkpoint['net']
        print("检测到包含'net'键的训练检查点，提取模型参数")
    else:
        state_dict = checkpoint
        print("加载标准模型状态字典")
    model_state_dict = model.state_dict()
    missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())
    unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())

    if missing_keys:
        print(f"警告: 缺少的键: {missing_keys}")
    if unexpected_keys:
        print(f"警告: 意外的键: {unexpected_keys}")

    matched_state_dict = {}
    for model_key, model_param in model_state_dict.items():
        if model_key in state_dict:
            pretrained_param = state_dict[model_key]
            if model_param.shape == pretrained_param.shape:
                matched_state_dict[model_key] = pretrained_param
            else:
                print(
                    f"跳过形状不匹配的参数: {model_key} 模型形状: {model_param.shape} 预训练形状: {pretrained_param.shape}")
        else:
            print(f"缺少参数: {model_key}")

    model.load_state_dict(matched_state_dict, strict=False)
    print(f"成功加载 {len(matched_state_dict)}/{len(model_state_dict)} 个参数")
    return model


def plot_tsne(features, labels, epoch, fold,cancer_type, save_dir):
    """t-SNE特征可视化（优化标签显示）"""
    tsne = TSNE(n_components=2, perplexity=15, learning_rate=200, n_iter=1000, random_state=42)
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(10, 8))

    # 动态获取唯一类别标签（避免硬编码）
    unique_labels = np.unique(labels)
    colors = ['dodgerblue', 'firebrick']  # 可扩展更多颜色应对多类别

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=colors[i], label=f'Class {label}', alpha=0.6, s=20
        )

    plt.title(f'Fold {fold + 1} Epoch {epoch + 1}: t-SNE of Latent Features', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.savefig(f'{save_dir}/fold_{fold + 1}_epoch_{epoch + 1}_{cancer_type}-_tsne.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def train_epoch(model, train_loaders, optimizer, criterion, vae_loss):
    """单轮训练函数（返回当前轮训练损失）"""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for data, labels in train_loaders:
        mixed_data, mixed_labels = data.to(device), labels.to(device).float()
        optimizer.zero_grad()

        mixed_labels = mixed_labels.unsqueeze(1)
        outputs = model(mixed_data)
        loss = criterion(outputs, mixed_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float().squeeze()
        correct += (preds == mixed_labels.squeeze()).sum().item()
        total += mixed_labels.size(0)

    accuracy = correct / total
    return train_loss / len(train_loaders), accuracy

    return train_loss / len(train_loaders)

def validate_epoch(model, test_loaders, criterion, device, epoch, fold, tsne_save_dir):
    """单轮验证函数（仅返回指定的核心指标）"""
    model.eval()
    val_loss = 0.0
    val_labels = []  # 真实标签
    val_probs = []  # 预测概率（用于计算阈值）
    all_features = []

    with torch.no_grad():
        for data, labels in test_loaders:
            data, labels = data.to(device), labels.to(device).float()
            outputs = model(data)

            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            val_labels.extend(labels.cpu().numpy().squeeze().astype(int))  # 真实标签
            val_probs.extend(torch.sigmoid(outputs).cpu().numpy().squeeze())  # 概率值

            h = model.encoder.backbone(data)
            z = model.encoder.projector(h)
            features = F.normalize(z, dim=1).cpu().numpy()
            all_features.append(features)

    val_labels = np.array(val_labels)
    val_probs = np.array(val_probs)

    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    optimal_idx = np.argmax(tpr - fpr)  # 最优阈值索引
    optimal_threshold = thresholds[optimal_idx]

    val_preds = (val_probs > optimal_threshold).astype(int)

    # 计算核心指标
    metrics = {
        'accuracy': accuracy_score(val_labels, val_preds),
        'precision': precision_score(val_labels, val_preds, zero_division=0),
        'recall': recall_score(val_labels, val_preds, zero_division=0),
        'f1_score': f1_score(val_labels, val_preds, zero_division=0),
        'auc': auc(fpr, tpr),  # AUC值
        'fpr': fpr,  # ROC曲线的FPR
        'tpr': tpr,  # ROC曲线的TPR
        'thresholds': thresholds,  # 所有阈值（用于后续分析）
        'val_labels': val_labels,  # 真实标签
        'val_preds': val_preds  # 基于最优阈值的预测标签
    }

    return val_loss / len(test_loaders), metrics

def save_metrics(all_metrics,mean_metrics,cancer_type,save_path='./metrics'):
    """保存所有fold的指标"""
    os.makedirs(save_path, exist_ok=True)
    long_format = []
    for fold_idx, fold_data in enumerate(all_metrics):
        for epoch, epoch_data in enumerate(fold_data):
            for metric, value in epoch_data.items():
                if metric not in ['fpr', 'tpr', 'thresholds','val_labels', 'val_preds']:  # 排除曲线数据
                    long_format.append({
                        'fold': fold_idx + 1,
                        'epoch': epoch + 1,
                        'metric': metric,
                        'value': value
                    })
    df = pd.DataFrame(long_format)
    df.to_csv(os.path.join(save_path, f'{cancer_type}_detailed_metrics.csv'), index=False)

    # 保存平均指标
    mean_df = pd.DataFrame([mean_metrics])
    mean_df.to_csv(os.path.join(save_path,f'{cancer_type}_mean_metrics.csv'), index=False)
    print(f"指标已保存至 {save_path}")

def plot_confusion_matrix(y_true,y_pred,fold,cancer_type,save_dir='./confusion_matrix'):
    """绘制混淆矩阵"""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1'],  # 根据实际类别命名
                yticklabels=['0', '1'])
    plt.title(f'Fold {fold + 1} Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.savefig(f'{save_dir}/fold_{fold + 1}_{cancer_type}_confusion_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot(all_metrics, all_fold_train_losses, all_fold_val_losses, mean_metrics,cancer_type, args):
    """绘制指标曲线"""
    epochs = range(1, args.num_epochs + 1)
    metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']

    for metric in metrics_list:
        plt.figure(figsize=(12, 8))
        for fold_idx in range(args.num_folds):
            metric_values = [all_metrics[fold_idx][epoch][metric] for epoch in range(args.num_epochs)]
            plt.plot(epochs, metric_values, label=f'Fold {fold_idx + 1}', marker='o',linestyle='-', linewidth=2)
        plt.title(f'{metric.capitalize()} Across All Folds', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(metric.capitalize(), fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(f'{cancer_type}_{metric}_trend.pdf')
        plt.show()
        plt.close()
    # 2. 损失趋势图
    plt.figure(figsize=(12, 8))
    for fold_idx in range(args.num_folds):
        plt.plot(epochs, all_fold_train_losses[fold_idx],
                 label=f'Fold {fold_idx + 1} Train Loss', linewidth=2)
        plt.plot(epochs, all_fold_val_losses[fold_idx],
                 label=f'Fold {fold_idx + 1} Val Loss', linewidth=2, linestyle='-')
    plt.title('Training & Validation Loss Across All Folds', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f'{cancer_type}_loss.pdf')
    plt.show()
    plt.close()

    # 3. 指标均值箱线图
    fold_metrics_means = []
    for fold_data in all_metrics:
        fold_mean = {metric: np.mean([epoch_data[metric] for epoch_data in fold_data]) for metric in metrics_list}
        fold_metrics_means.append(fold_mean)

    # 整理为箱线图所需的格式：{指标: [折1均值, 折2均值, ...]}
    final_metrics_mean = {
        metric: [fold_mean[metric] for fold_mean in fold_metrics_means]
        for metric in metrics_list
    }

    plt.figure(figsize=(10, 6))
    boxplot = plt.boxplot(final_metrics_mean.values(), labels=metrics_list, patch_artist=True)
    # 美化箱线图
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
    plt.title('Mean Metrics Across All Folds (Each Fold\'s Average)', fontsize=16)
    plt.ylabel('Mean Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xticks(rotation=45)
    plt.savefig(f'{cancer_type}_mean_metrics_boxplot.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # 4. ROC曲线对比图
    plt.figure(figsize=(10, 8))
    for fold_idx in range(args.num_folds):
        final_epoch_data = all_metrics[fold_idx][-1]
        plt.plot(final_epoch_data['fpr'], final_epoch_data['tpr'], lw=2,
                 label=f'Fold {fold_idx + 1} (AUC={final_epoch_data["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves of All Folds', fontsize=16)
    plt.legend(loc='lower right')
    plt.savefig(f'{cancer_type}_final_roc_curves.pdf')
    plt.show()
    plt.close()

def main(args):
    tsne_save_dir = './tsne_visualizations'
    metrics_save_dir = './metrics'
    confusion_matrix_dir = './confusion_matrices'
    os.makedirs(tsne_save_dir, exist_ok=True)
    os.makedirs(metrics_save_dir, exist_ok=True)
    os.makedirs(confusion_matrix_dir, exist_ok=True)

    view_list = ['rnaseq', 'mirna', 'psi', 'cnv', 'dnaMeth']
    (all_train_loaders, all_test_loaders, all_ds) = data_loader(
        cancer_type=args.cancer_type,
        batch_size=args.batch_size,
        seed=args.seed,
        view_list=view_list,
        device=device,
    )

    all_metrics = []  # 存储所有fold的所有epoch指标
    all_fold_train_losses = []  # 存储各fold的训练损失
    all_fold_val_losses = []  # 存储各fold的验证损失
    all_fold_train_accs = []
    for fold in range(args.num_folds):

        mveb = MVEB(all_ds[fold], latent_dim=args.latent_dim, proj_dim=args.proj_dim).to(device)
        mveb = load_pretrained_model(mveb, args.pretrained_path, device)
        mveb.eval()
        model = MVEBClassifier(mveb, num_classes=1, freeze_encoder=args.freeze_encoder).to(device)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        # 损失函数
        criterion = nn.BCEWithLogitsLoss().to(device)
       # criterion = FocalLoss(alpha=0.8, gamma=2).to(device)
        logger = SummaryWriter(log_dir=f"./logs/fold_{fold + 1}")

        fold_metrics = []
        fold_train_losses = []
        fold_val_losses = []
        fold_train_accs = []
        print(f"\n===== Starting Fold {fold + 1} =====")
        for epoch in range(args.num_epochs):

            train_loss,train_acc = train_epoch(model, all_train_loaders[fold],
                optimizer,
                criterion,
                device)
            fold_train_losses.append(train_loss)  # 记录训练损失
            fold_train_accs.append(train_acc)
            val_loss, metrics = validate_epoch(model,
                all_test_loaders[fold],
                criterion,
                device,
                epoch,
                fold,
                tsne_save_dir)
            scheduler.step(metrics['auc'])
            fold_val_losses.append(val_loss)  # 记录验证损失
            fold_metrics.append(metrics)


            if epoch % 20 == 0:
                print(f"Epoch {epoch + 1}/{args.num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {metrics['accuracy']:.4f} | "
                      f"AUC: {metrics['auc']:.4f}| F1: {metrics['f1_score']:.4f}|recall: {metrics['recall']:.4f}| precision: {metrics['precision']:.4f}|")
            logger.add_scalar('train_loss', train_loss, epoch)
            logger.add_scalar('val_loss', val_loss, epoch)
            for metric, value in metrics.items():
                if metric not in ['fpr', 'tpr', 'thresholds', 'val_labels', 'val_preds']:
                    logger.add_scalar(f'val_{metric}', value, epoch)
        all_metrics.append(fold_metrics)
        all_fold_train_losses.append(fold_train_losses)
        all_fold_val_losses.append(fold_val_losses)
        all_fold_train_accs.append(fold_train_accs)

        # 绘制混淆矩阵
        final_metrics = fold_metrics[-1]  # 取最后一个epoch的指标
        plot_confusion_matrix(final_metrics['val_labels'], final_metrics['val_preds'], fold,args.cancer_type, confusion_matrix_dir)

        logger.close()

    mean_final_metrics = {
        'accuracy': np.mean([fold[-1]['accuracy'] for fold in all_metrics]),
        'precision': np.mean([fold[-1]['precision'] for fold in all_metrics]),
        'recall': np.mean([fold[-1]['recall'] for fold in all_metrics]),
        'f1_score': np.mean([fold[-1]['f1_score'] for fold in all_metrics]),
        'auc': np.mean([fold[-1]['auc'] for fold in all_metrics])
    }
    save_metrics(all_metrics, mean_final_metrics, metrics_save_dir)

    plot(all_metrics, all_fold_train_losses, all_fold_val_losses, mean_final_metrics,args.cancer_type, args)
    print(f"最终平均指标: {mean_final_metrics}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    parser.add_argument('--cancer_type', type=str, default='KIRC', help='cancer type')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight_decay')
    parser.add_argument('--num_epochs', type=int, default=100, help='num_epochs')
    parser.add_argument('--num_folds', type=int, default=3, help='folds')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension size')
    parser.add_argument('--proj_dim', type=int, default=100, help='Projection dimension size')
    parser.add_argument('--classifier_size', type=int, default=8, help='classifier input size')
    parser.add_argument('--pretrained_path', type=str, default='./models/vae_0.0001_32_100.pth',
                        help='path to pretrained VAE model')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='是否冻结MVEB编码器的参数')
    config = yaml_config_hook("config/config.yaml")
    for key, value in config.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    main(args)
