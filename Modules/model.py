import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from pretrain_vae.vae_models import  MVEB
class MVEBClassifier(nn.Module):
    def __init__(self, mveb_model, num_classes, freeze_encoder=True):
        """
        MVEB模型 + 线性分类器

        Args:
            mveb_model: 预训练的MVEB模型
            num_classes: 分类类别数
            freeze_encoder: 是否冻结MVEB编码器
        """
        super().__init__()
        self.encoder = mveb_model
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.projector.project[0].in_features, 50),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(50, num_classes))

        # 冻结编码器参数
        if freeze_encoder:
           # for param in self.encoder.parameters():
           #     param.requires_grad = False
           # 冻结 backbone 的前几层（保留底层特征）
           for i, layer in enumerate(self.encoder.backbone.backbone):
               if i < 4:  # 只冻结前4层（可根据需要调整）
                   for param in layer.parameters():
                       param.requires_grad = False
               else:
                   for param in layer.parameters():
                       param.requires_grad = True

    def forward(self, x):
        # 获取MVEB表示
        with torch.set_grad_enabled(not self.encoder.training):
            features = self.encoder.backbone(x)

        # 分类预测
        logits = self.classifier(features)
        return logits
