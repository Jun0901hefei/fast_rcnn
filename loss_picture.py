import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.ops as ops
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt  # 添加 matplotlib 用于绘图
import os  # 添加 os 用于文件操作

class LossHistory:
    """
    损失历史记录类，用于保存和可视化训练损失
    """
    def __init__(self, save_dir='loss_history'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        # 存储损失值
        self.epochs = []
        self.train_loss = []
        self.train_cls_loss = []
        self.train_bbox_loss = []
        self.batch_loss = []  # 每个batch的损失
        self.batch_indices = []  # batch索引
        self.batch_cls_loss = []  # 每个batch的分类损失
        self.batch_bbox_loss = []  # 每个batch的回归损失

    def add_epoch_loss(self, epoch, loss, cls_loss, bbox_loss):
        """添加每个epoch的损失"""
        self.epochs.append(epoch)
        self.train_loss.append(loss)
        self.train_cls_loss.append(cls_loss)
        self.train_bbox_loss.append(bbox_loss)

    def add_batch_loss(self, batch_idx, loss, cls_loss, bbox_loss):
        """添加每个batch的损失"""
        self.batch_indices.append(batch_idx)
        self.batch_loss.append(loss)
        self.batch_cls_loss.append(cls_loss)
        self.batch_bbox_loss.append(bbox_loss)

    def plot_loss(self, show=True, save=True):
        """
        绘制损失曲线
        4个子图：
        1. 总损失 vs Epoch
        2. 分类损失 vs Epoch
        3. 回归损失 vs Epoch
        4. 所有Batch损失（包含总损失、分类损失、回归损失的对比）
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # 1. 总损失 (每个epoch)
        ax1 = axes[0, 0]
        if self.epochs:
            ax1.plot(self.epochs, self.train_loss, 'b-o', linewidth=2, markersize=6)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Total Loss', fontsize=12)
            ax1.set_title('Total Loss vs Epoch', fontsize=14)
            ax1.grid(True, alpha=0.3)
        # 2. 分类损失 (每个epoch)
        ax2 = axes[0, 1]
        if self.epochs:
            ax2.plot(self.epochs, self.train_cls_loss, 'r-o', linewidth=2, markersize=6, label='Classification Loss')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Classification Loss', fontsize=12)
            ax2.set_title('Classification Loss vs Epoch', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        # 3. 回归损失 (每个epoch)
        ax3 = axes[1, 0]
        if self.epochs:
            ax3.plot(self.epochs, self.train_bbox_loss, 'g-o', linewidth=2, markersize=6, label='BBox Regression Loss')
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('BBox Regression Loss', fontsize=12)
            ax3.set_title('BBox Regression Loss vs Epoch', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        # 4. 所有Batch损失（包含三种损失）
        ax4 = axes[1, 1]
        if self.batch_indices and len(self.batch_indices) > 0:
            # 总损失（原始点）
            ax4.plot(self.batch_indices, self.batch_loss, 'b.', alpha=0.5, markersize=3, label='Total Loss')
            # 分类损失（原始点）
            ax4.plot(self.batch_indices, self.batch_cls_loss, 'r.', alpha=0.5, markersize=3, label='Cls Loss')

            # 回归损失（原始点）
            ax4.plot(self.batch_indices, self.batch_bbox_loss, 'g.', alpha=0.5, markersize=3, label='BBox Loss')
            ax4.set_xlabel('Batch Iteration (累计batch数)', fontsize=12)
            ax4.set_ylabel('Loss', fontsize=12)
            ax4.set_title('All Batch Losses (详细训练过程)', fontsize=14, fontweight='bold')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            save_path = os.path.join(self.save_dir, 'loss_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Loss curves saved to {save_path}')
        if show:
            plt.show()
        else:
            plt.close()

    def save_data(self):
        """保存损失数据到CSV"""
        data = {
            'epoch': self.epochs,
            'train_loss': self.train_loss,
            'train_cls_loss': self.train_cls_loss,
            'train_bbox_loss': self.train_bbox_loss,
            'batch_indices': self.batch_indices,
            'batch_loss': self.batch_loss,
            'batch_cls_loss': self.batch_cls_loss,
            'batch_bbox_loss': self.batch_bbox_loss
        }
        df = pd.DataFrame(data)
        save_path = os.path.join(self.save_dir, 'loss_data.csv')
        df.to_csv(save_path, index=False)
        print(f'Loss data saved to {save_path}')