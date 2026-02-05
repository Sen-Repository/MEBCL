import argparse
import torch
import torch.nn.functional as F
from torch import nn
from data_processing import multi_cancer_data_loader
from utils import yaml_config_hook
from vae_models import MVEB, mveb_loss
#from model import model
from utils import save_model
import matplotlib.pyplot as plt

def add_noise_views(x, noise_std=0.1, device=None):

    noise1 = torch.randn_like(x, device=device) * noise_std
    noise2 = torch.randn_like(x, device=device) * noise_std
    return x + noise1, x + noise2

def train_ae_vicreg(model, loader, optimizer, device, noise_std=0.1, lambda_entropy=1.0, kernel_matrix=None):

    model.train()
    total_stats = {'align_loss': 0, 'entropy_z1': 0, 'entropy_z2': 0,
                   'entropy_loss': 0, 'total_loss': 0}

    for batch in loader:

       v1, v2 = add_noise_views(batch, noise_std=noise_std, device=device)

       z1 = model(v1)
       z2 = model(v2)

       loss, stats = mveb_loss(z1, z2, lambda_entropy=lambda_entropy)

       optimizer.zero_grad()
       loss.backward()
       nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
       optimizer.step()

       for k in total_stats:
            total_stats[k] += stats[k]

    num_batches = len(loader)
    for k in total_stats:
        total_stats[k] /= num_batches if num_batches > 0 else 1

    return total_stats['total_loss'], total_stats

def evaluate_ae_vicreg(model, loader, device, noise_std=0.1,lambda_entropy=1.0):

    model.eval()
    total_stats = {'align_loss': 0, 'entropy_z1': 0, 'entropy_z2': 0,
                   'entropy_loss': 0, 'total_loss': 0}

    with torch.no_grad():
        for batch in loader:
            v1, v2 = add_noise_views(batch, noise_std=noise_std, device=device)
            z1 = model(v1)
            z2 = model(v2)
            loss, stats = mveb_loss(z1, z2, lambda_entropy=lambda_entropy)

            for k in total_stats:
                total_stats[k] += stats[k]

    num_batches = len(loader)
    for k in total_stats:
        total_stats[k] /= num_batches if num_batches > 0 else 1

    return total_stats['total_loss'], total_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 基础参数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--proj_dim", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--noise_std", type=float, default=0.1, help="高斯噪声标准差")
    parser.add_argument("--beta", type=float, default=1.0, help="熵损失权重")

    args = parser.parse_args()

    # 加载配置有
    config = yaml_config_hook("../config/config.yaml")
    for key, value in config.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()

    # 加载数据（移除交叉验证相关返回值）
    cancer_types = config.get('cancer_types', [])
    view_list = config.get('view_list', [])
    seed = args.seed if hasattr(args, 'seed') else 42

    # 注意：这里只保留完整训练集和测试集加载器
    (full_train_loader, test_loader, feature_dim) = multi_cancer_data_loader(
        cancer_types=cancer_types,
        batch_size=args.batch_size,
        seed=seed,
        view_list=view_list,
        device=device,
        fillna_value=0
    )

    # 初始化模型（使用训练集特征维度）
    model = MVEB(feature_dim, args.latent_dim,args.proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    # 训练与验证

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        # 训练
        train_total, train_stats = train_ae_vicreg(
            model, full_train_loader, optimizer, device
        )
        train_losses.append(train_stats['total_loss'])
        val_total, val_stats = evaluate_ae_vicreg(
            model, test_loader, device
        )
        val_losses.append(val_stats['total_loss'])
        print(f"Epoch {epoch + 1:3d}/{args.epochs} | "
              f"Train: Total {train_total:.4f} (Align {train_stats['align_loss']:.4f}, Ent {train_stats['entropy_loss']:.4f}/{train_stats['entropy_z2']:.4f}) | "
              f"Val:   Total {val_total:.4f} (Align {val_stats['align_loss']:.4f}, Ent {val_stats['entropy_loss']:.4f}/{val_stats['entropy_z2']:.4f})")
        
    # 保存模型
    save_model(model, optimizer, args.epochs, f"../models/vae_{args.learning_rate}_{args.batch_size}_{args.epochs}.pth")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(range(1, args.epochs + 1), val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14)
    plt.legend(fontsize=12)
    plt.xlim(1, args.epochs)
    plt.tight_layout()
    plt.savefig('Evualuation.pdf')
    plt.show()

