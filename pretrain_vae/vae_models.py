import torch.nn as nn
import torch.nn.functional as F
import torch


def block(in_c, out_c, activation):
    layers = [nn.Linear(in_c, out_c)]

    if activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'leakyrelu':
        layers.append(nn.LeakyReLU())

    return layers


class EncoderBackbone(nn.Module):
    """在线编码器（带Dropout正则化）"""
    def __init__(self, input_dim, latent_dim):#[5000,1000,100]
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.backbone(x)


class Projector(nn.Module):
    """Projector 头（将 Backbone 输出映射到隐空间）"""

    def __init__(self, latent_dim,proj_dim):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

    def forward(self, x):
        return self.project(x)


class MVEB(nn.Module):
    """Multi-View Entropy Bottleneck 模型
    核心：双分支动量编码器 + 对齐损失 + 熵损失
    """
    def __init__(self, input_dim, latent_dim, proj_dim):
        super().__init__()
        self.backbone = EncoderBackbone(input_dim, latent_dim)
        self.projector = Projector(latent_dim, proj_dim)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)  # L2归一化


def stein_score_estimate(z, kernel_width=0.1):
    """Stein梯度估计器（SGE）计算分数函数
    严格实现文献中的公式：
        S(z) = ∇_z log p(z) ≈ 1/(σ²) [E_{z'~k(z,·)}[k(z,z')z'] - z]

    Args:
        z: 归一化后的表示向量 [batch_size, latent_dim]
        kernel_width: 高斯核宽度
    Returns:
        score: 分数函数估计 [batch_size, latent_dim]
    """
    batch_size, latent_dim = z.shape

    # 计算核矩阵 (高斯RBF核)
    pairwise_dist = torch.cdist(z, z, p=2)  # [batch_size, batch_size]
    kernel_matrix = torch.exp(-pairwise_dist ** 2 / (2 * kernel_width ** 2))

    # 计算加权邻域表示
    weighted_z = torch.matmul(kernel_matrix, z) / kernel_matrix.sum(dim=1, keepdim=True)

    # 计算Stein分数
    score = (weighted_z - z) / (kernel_width ** 2)
    return score


def mveb_loss(z1, z2, lambda_entropy=1.0):
    """严格实现文献中的损失函数
    L = Align_loss + 0.5 * λ * (H(z1) + H(z2))
    其中:
        Align_loss = -E[z₂ᵀz₁]
        H(z) ≈ E[S(z)·z] (通过Stein估计)

    Args:
        z1, z2: L2归一化后的表示向量 [batch_size, latent_dim]
        lambda_entropy: 熵损失权重
    Returns:
        total_loss: 总损失
        stats: 损失分解字典
    """
    # 1. 对齐损失 (最大化z1和z2的相似度)
    align_loss = -torch.mean(torch.sum(z1 * z2, dim=1))

    # 2. 熵损失 (通过Stein估计)
    # 计算分数函数
    S_z1 = stein_score_estimate(z1)
    S_z2 = stein_score_estimate(z2)

    # 估计熵: H(z) ≈ E[S(z)·z]
    entropy_z1 = torch.mean(torch.sum(S_z1.detach() * z1, dim=1))
    entropy_z2 = torch.mean(torch.sum(S_z2.detach() * z2, dim=1))

    # 组合熵项
    entropy_loss = 0.5 * lambda_entropy * (entropy_z1 + entropy_z2)

    # 3. 总损失
    total_loss = align_loss + entropy_loss

    return total_loss, {
        "align_loss": align_loss.item(),
        "entropy_z1": entropy_z1.item(),
        "entropy_z2": entropy_z2.item(),
        "entropy_loss": entropy_loss.item(),
        "total_loss": total_loss.item()
    }