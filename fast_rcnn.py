import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.ops as ops
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple
import cv2
from selective_search import SelectiveSearch
import anchor_box_generator
import assign_anchor_box
import corner_and_center
import IOU
import Mark_categories_and_offsets
import NMSandMulti_box_detection
import predict
import data_precession
class FastRCNN(nn.Module):
    """Fast R-CNN模型"""
    def __init__(self, num_classes=21, roi_output_size=(7, 7),
                 ss_mode='fast', max_proposals=2000):
        super(FastRCNN, self).__init__()
        self.num_classes = num_classes
        self.roi_output_size = roi_output_size
        self.max_proposals = max_proposals

        # Selective Search
        self.ss = SelectiveSearch(mode=ss_mode, max_rois=max_proposals)
        #在imagenet上与训练的vgg
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # 提取VGG16的卷积层部分（不包括全连接层）
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:30])
        # VGG16有4个池化层，所以缩放比例为1/16
        self.num_pooling_layers = 4
        self.spatial_scale = 1.0 / (2 ** self.num_pooling_layers)  # 1/16
        # 获取特征图的通道数
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.feature_extractor(dummy_input)
            self.feature_channels = dummy_features.size(1)#特征图的输出通道数
        # 使用torchvision.ops.RoIPool作为RoI Pooling层
        self.roi_pool = ops.RoIPool(output_size=roi_output_size,
                                    spatial_scale=self.spatial_scale)
        # 全连接层
        fc_input_size = self.feature_channels * roi_output_size[0] * roi_output_size[1]
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        # 分类和回归头
        self.classifier = nn.Linear(256, num_classes)
        self.bbox_regressor = nn.Linear(256, num_classes * 4)
        # 初始化权重
        self._initialize_weights()
    def _initialize_weights(self):
        """初始化全连接层权重"""
        for layer in [self.fc1, self.fc2, self.classifier, self.bbox_regressor]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x, proposals=None):
        """
        前向传播
            x: 输入图像 (batch_size, 3, H, W)
            proposals: 可选，预先生成的候选区域
        Returns:
            cls_scores_list: 分类分数列表，当前batch每张图片的类别分数，每个元素为 (num_rois, num_classes)
            bbox_deltas_list: 边界框回归偏移列表，当前batch每张图片的类别偏移量，每个元素为 (num_rois, num_classes * 4)
        """
        batch_size = x.size(0)
        device = x.device
        # 步骤1: 将整张图片导入VGG中，输出卷积层输出的feature map
        # 输入: (batch_size, 3, H, W) -> 输出: (batch_size, 512, H/16, W/16)
        features = self.feature_extractor(x)
        # 步骤2: 准备RoIs (使用原图坐标)
        if proposals is None:
            all_proposals = []
            for i in range(batch_size):
                img = x[i]  # (3, H, W)
                proposals_i = self.generate_proposals(img)
                all_proposals.append(proposals_i)
                # 合并所有候选区域
                # 注意：不同图像可能有不同数量的候选区域
            max_proposals = max([p.size(0) for p in all_proposals])
            # Padding 到相同数量
            padded_proposals = []
            for p in all_proposals:
                if p.size(0) < max_proposals:
                    pad = torch.zeros(max_proposals - p.size(0), 4, device=device)
                    p = torch.cat([p, pad], dim=0)
                padded_proposals.append(p)

            proposals = torch.stack(padded_proposals, dim=0)  # (batch_size, max_proposals, 4)
        num_proposals = proposals.size(1)
        proposals_flat = proposals.reshape(-1, 4)  # (batch_size * num_proposals, 4)
        # 步骤3: 为所有batch的RoI添加batch索引（批量处理）
        # 为每个batch创建对应的batch索引
        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(num_proposals).unsqueeze(1)
        # batch_indices: (batch_size * num_proposals, 1)

        rois_with_batch = torch.cat([batch_indices, proposals_flat], dim=1)
        # (batch_size * num_proposals, 5) [batch_idx, x1, y1, x2, y2]
        # 步骤4: 一次性进行RoI Pooling（批量处理）
        pooled = self.roi_pool(features, rois_with_batch)
        # pooled: (batch_size * num_rois, C, 7, 7)
        # 步骤5: 展平(batch_size * num_rois, C * 7 * 7)
        flattened = pooled.view(pooled.size(0), -1)
        # 步骤6: 导入全连接层，输出 (batch_size * num_rois, 4096) 维的张量
        fc1_out = F.relu(self.fc1(flattened))  # (batch_size * num_rois, 4096)
        fc2_out = F.relu(self.fc2(fc1_out))  # (batch_size * num_rois, 4096)
        # 步骤7: 将这个张量导入分类器和回归器
        # 分类: (batch_size * num_rois, num_classes)
        cls_scores = self.classifier(fc2_out)
        # 回归: (batch_size * num_rois, num_classes * 4)
        bbox_deltas = self.bbox_regressor(fc2_out)
        # 步骤8: 按batch分割结果
        cls_scores_list = []
        bbox_deltas_list = []
        for i in range(batch_size):
            start_idx = i * num_proposals
            end_idx = (i + 1) * num_proposals
            cls_scores_list.append(cls_scores[start_idx:end_idx])
            bbox_deltas_list.append(bbox_deltas[start_idx:end_idx])
        return cls_scores_list, bbox_deltas_list


    def generate_proposals(self, image_tensor):
        """
        从图像张量生成候选区域
        Args:
            image_tensor: (3, H, W) 张量，值域 [0, 1]
        Returns:
            proposals: (N, 4) 候选区域 [x1, y1, x2, y2]
        """
        # 转换为 numpy (H, W, 3)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

        # 生成候选区域
        proposals, _ = self.ss.generate_proposals(image_np)

        return torch.tensor(proposals, dtype=torch.float32, device=image_tensor.device)

