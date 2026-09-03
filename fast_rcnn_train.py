import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.ops as ops
from matplotlib import patches
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt  # 添加 matplotlib 用于绘图
import os
import anchor_box_generator
import assign_anchor_box
import corner_and_center
import IOU
import Mark_categories_and_offsets
import NMSandMulti_box_detection
import predict
import data_precession
import fast_rcnn
import loss_picture

class FastRCNNTrainer:
    """
    Fast R-CNN训练器
    """
    def __init__(self, model, device='cuda', iou_threshold=0.2, loss_history=None):
        self.model = model
        self.device = device
        self.iou_threshold = iou_threshold
        self.loss_history = loss_history  # 添加损失历史记录
        self.global_batch_idx = 0  # 全局累计batch计数器
    def prepare_targets(self, proposals_batch, labels):
        """
        准备训练目标（使用候选区域而不是锚框）

        Args:
            proposals_batch: (batch_size, num_proposals, 4) 候选区域坐标
            labels: (batch_size, num_gt_boxes, 5) 真实标签
        Returns:
            bbox_offset: (batch_size, num_proposals * 4)
            bbox_mask: (batch_size, num_proposals * 4)
            class_labels: (batch_size, num_proposals)
        """
        return Mark_categories_and_offsets.multi_box_target(proposals_batch, labels)

    def train_step(self, images, labels, optimizer):
        """
        单步训练，训练一个batch
        """
        self.model.train()
        images = images.to(self.device)
        # 处理labels维度
        if labels.dim() == 2:
            # (batch_size, 5) -> (batch_size, 1, 5)
            labels = labels.unsqueeze(1)
        elif labels.dim() == 1:
            # (5,) -> (1, 1, 5)
            labels = labels.unsqueeze(0).unsqueeze(0)
        labels = labels.to(self.device)

        batch_size, channel, h, w = images.shape
        with torch.no_grad():
            all_proposals = []
            for i in range(batch_size):
                img = images[i]  # (3, H, W)
                proposals_i = self.model.generate_proposals(img)
                all_proposals.append(proposals_i)

                # 处理不同数量的候选区域
            max_proposals = max([p.size(0) for p in all_proposals])
            if max_proposals == 0:
                # 如果没有候选区域，返回零损失
                return 0.0, 0.0, 0.0

            # Padding 到相同数量
            padded_proposals = []
            for p in all_proposals:
                if p.size(0) < max_proposals:
                    pad = torch.zeros(max_proposals - p.size(0), 4, device=self.device)
                    p = torch.cat([p, pad], dim=0)
                padded_proposals.append(p)

            proposals_batch = torch.stack(padded_proposals, dim=0)  # (batch_size, max_proposals, 4)

            #bbox_offset：形状(batch_size, num_anchors * 4)
            #bbox_mask：形状(batch_size, num_anchors * 4)
            #class_labels：形状(batch_size, num_anchors)
            bbox_offset, bbox_mask, class_labels = self.prepare_targets(proposals_batch, labels)

        # 前向传播 (传入原图坐标的锚框)
        cls_scores_list, bbox_deltas_list = self.model(images, proposals_batch)

        # 将列表转换为张量（批量处理）
        cls_scores = torch.stack(cls_scores_list, dim=0)  # (batch_size, num_anchors, num_classes)
        bbox_deltas = torch.stack(bbox_deltas_list, dim=0)  # (batch_size, num_anchors, num_classes * 4)
        # 立即释放列表中的张量
        del cls_scores_list, bbox_deltas_list
        # 重塑 bbox_deltas 为 (batch_size, num_anchors, num_classes, 4)
        num_anchors = bbox_deltas.size(1)
        num_classes = self.model.num_classes
        bbox_deltas_reshaped = bbox_deltas.view(batch_size, num_anchors, num_classes, 4)

        # 获取真实标签和掩码
        target_labels = class_labels  # (batch_size, num_anchors)
        target_offsets = bbox_offset.reshape(batch_size, num_anchors, 4)  # (batch_size, num_anchors, 4)
        target_mask = bbox_mask.reshape(batch_size, num_anchors, 4)  # (batch_size, num_anchors, 4)

        # ============ 批量计算分类损失 ============
        # 重塑分类分数和标签
        cls_scores_flat = cls_scores.reshape(-1, num_classes)  # (batch_size * num_anchors, num_classes)
        target_labels_flat = target_labels.reshape(-1)  # (batch_size * num_anchors,)

        # 分类损失（批量计算）
        cls_loss = F.cross_entropy(cls_scores_flat, target_labels_flat, reduction='mean')

        # ============ 批量计算回归损失 ============
        # 获取正样本掩码 (batch_size, num_anchors)
        pos_mask = target_mask[:, :, 0] > 0  # (batch_size, num_anchors)

        # 如果没有正样本，回归损失为0
        if pos_mask.sum() == 0:
            bbox_loss = torch.tensor(0.0, device=self.device)
        else:
            # 获取正样本索引
            pos_indices = torch.nonzero(pos_mask)  # (num_pos, 2) 每行是 [batch_idx, anchor_idx]
            num_pos = pos_indices.size(0)
            # 提取正样本的类别标签
            pos_target_labels = target_labels[pos_indices[:, 0], pos_indices[:, 1]]  # (num_pos,)
            # 提取正样本对应的回归值（对应类别的4个值）
            pos_bbox_deltas = bbox_deltas_reshaped[
                              pos_indices[:, 0], pos_indices[:, 1], pos_target_labels, :
                              ]  # (num_pos, 4)
            # 提取正样本的真实偏移量
            pos_target_offsets = target_offsets[pos_indices[:, 0], pos_indices[:, 1], :]  # (num_pos, 4)
            # 计算回归损失（Smooth L1 Loss，使用sum然后平均）
            bbox_loss = F.smooth_l1_loss(pos_bbox_deltas, pos_target_offsets, reduction='sum') / num_pos
            # 释放中间变量
            del pos_indices, pos_target_labels, pos_bbox_deltas, pos_target_offsets
        # 总损失
        total_loss = cls_loss + bbox_loss
        # 释放大张量（但保留 total_loss 用于反向传播）
        del cls_scores, bbox_deltas, bbox_deltas_reshaped
        del target_labels, target_offsets, target_mask
        del cls_scores_flat, target_labels_flat, pos_mask
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        loss_value = total_loss.item()
        cls_loss_value = cls_loss.item()
        bbox_loss_value = bbox_loss.item()
        del total_loss, cls_loss, bbox_loss
        return loss_value, cls_loss_value, bbox_loss_value
    def train_epoch(self, dataloader, optimizer, epoch_idx):
        """
        训练一个epoch
        """
        total_loss = 0
        total_cls_loss = 0
        total_bbox_loss = 0
        # 创建tqdm进度条
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch_idx}",
            dynamic_ncols=True
        )
        for batch_idx, (images, labels) in  pbar:
            # 使用累计的全局batch编号
            current_batch_idx = self.global_batch_idx
            loss, cls_loss, bbox_loss = self.train_step(images, labels, optimizer)
            total_loss += loss
            total_cls_loss += cls_loss
            total_bbox_loss += bbox_loss
            # 记录batch损失（使用累计编号）
            if self.loss_history:
                self.loss_history.add_batch_loss(current_batch_idx, loss, cls_loss, bbox_loss)
            # 全局batch计数器递增
            self.global_batch_idx += 1
            # 更新进度条显示，在右侧增加额外信息
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'cls': f'{cls_loss:.4f}',
                'bbox': f'{bbox_loss:.4f}'
            })
        avg_loss = total_loss / len(dataloader)
        avg_cls_loss = total_cls_loss / len(dataloader)
        avg_bbox_loss = total_bbox_loss / len(dataloader)
        # 记录epoch损失
        if self.loss_history:
            self.loss_history.add_epoch_loss(epoch_idx, avg_loss, avg_cls_loss, avg_bbox_loss)
            # 每次epoch结束后保存数据和绘制图表
            self.loss_history.save_data()
            self.loss_history.plot_loss(show=False, save=True)
        return avg_loss, avg_cls_loss, avg_bbox_loss

    def train(self, train_dataloader, num_epochs=10, learning_rate=0.00001,
              momentum=0.9, weight_decay=0.0005, save_dir='checkpoints'):
        """
        完整训练流程
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 优化器
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                    momentum=momentum, weight_decay=weight_decay)
        print(f'Starting training for {num_epochs} epochs...')
        print(f'Model device: {self.device}')
        print(f'Number of batches: {len(train_dataloader)}')
        for epoch in range(1, num_epochs + 1):
            print(f'\nEpoch {epoch}/{num_epochs}')
            train_loss, train_cls_loss, train_bbox_loss = self.train_epoch(
                train_dataloader, optimizer, epoch
            )
            print(f'\nEpoch {epoch} Summary:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Cls Loss: {train_cls_loss:.4f}')
            print(f'  Bbox Loss: {train_bbox_loss:.4f}')
        # 最终保存
        model_path = os.path.join(save_dir, 'fastrcnn.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f'\nModel saved: {model_path}')
        # 显示最终的损失曲线
        if self.loss_history:
            print('\nDisplaying final loss curves...')
            self.loss_history.plot_loss(show=True, save=True)
        print('\nTraining completed!')

    @torch.no_grad()
    def predict(self, images, nms_threshold=0.5, pos_threshold=0.5, keep_top_per_class=True):
        """
        预测一个batch
        :param images: 一个batch的图片
        :param nms_threshold: NMS阈值
        :param pos_threshold: 置信度阈值
        :param keep_top_per_class: 是否保留每个类别置信度最高的框
        :return: 一个列表，每个元素储存当前图片的预测结果 [num_anchors, 6]
                 如果keep_top_per_class为True，则只保留每个类别置信度最高的框[batch_size,num_class,6]
        """
        self.model.eval()
        images = images.to(self.device)
        batch_size, channel, h, w = images.shape
        # 使用 Selective Search 生成候选区域
        all_proposals = []
        for i in range(batch_size):
            img = images[i]
            proposals_i = self.model.generate_proposals(img)
            all_proposals.append(proposals_i)
        # 处理不同数量的候选区域
        max_proposals = max([p.size(0) for p in all_proposals])
        if max_proposals == 0:
            return [torch.zeros(0, 6, device=self.device) for _ in range(batch_size)]

        padded_proposals = []
        for p in all_proposals:
            if p.size(0) < max_proposals:
                pad = torch.zeros(max_proposals - p.size(0), 4, device=self.device)
                p = torch.cat([p, pad], dim=0)
            padded_proposals.append(p)

        proposals_batch = torch.stack(padded_proposals, dim=0)
        # 前向传播
        cls_scores_list, bbox_deltas_list = self.model(images, proposals_batch)
        # 将列表转换为张量（批量处理）
        cls_scores = torch.stack(cls_scores_list, dim=0)  # (batch_size, num_anchors, num_classes)
        bbox_deltas = torch.stack(bbox_deltas_list, dim=0)  # (batch_size, num_anchors, num_classes * 4)
        # 如果没有任何锚框，返回空结果
        if cls_scores.size(1) == 0:
            return [torch.zeros(0, 6, device=self.device) for _ in range(batch_size)]
        # Softmax 得到概率
        cls_probs = F.softmax(cls_scores, dim=2)  # (batch_size, num_anchors, num_classes)
        # 获取每个锚框的预测类别
        pred_class_ids = torch.argmax(cls_probs, dim=2)  # (batch_size, num_anchors)
        # 重塑 bbox_deltas 为 (batch_size, num_anchors, num_classes, 4)
        num_anchors = bbox_deltas.size(1)
        num_classes = self.model.num_classes
        bbox_deltas_reshaped = bbox_deltas.view(batch_size, num_anchors, num_classes, 4)
        # 提取每个锚框对应预测类别的回归值
        #[[0,0,0,0,0,0],
        # [1,1,1,1,1,1],
        # [2,2,2,2,2,2],
        # ....]
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, num_anchors)#(batch,anchor)
        # [[0,1,2,3,4,5],
        #  [0,1,2,3,4,5],
        #  [0,1,2,3,4,5],
        # ....]
        anchor_indices = torch.arange(num_anchors, device=self.device).unsqueeze(0).expand(batch_size, -1)#(batch,anchor)
        # selected_bbox_deltas: (batch_size, num_anchors, 4)
        selected_bbox_deltas = bbox_deltas_reshaped[batch_indices, anchor_indices, pred_class_ids, :]
        # 转换为 multi_box_detection 需要的格式
        # cls_probs: (batch_size, num_classes, num_anchors)
        cls_probs_transposed = cls_probs.permute(0, 2, 1)  # (batch_size, num_classes, num_anchors)
        bbox_deltas_flat = selected_bbox_deltas.reshape(batch_size, -1)  # (batch_size, num_anchors * 4)
        # 批量 NMS
        detections_batch = NMSandMulti_box_detection.multi_box_detection(
            cls_probs_transposed,
            bbox_deltas_flat,
            proposals_batch,
            nms_threshold=nms_threshold,
            pos_threshold=pos_threshold
        )
        # detections_batch: (batch_size, num_anchors, 6)
        # 分割结果
        results = []
        for i in range(batch_size):
            detections = detections_batch[i]  # (num_anchors, 6)
            if keep_top_per_class:
                # 保留每个类别置信度最高的框
                detections = self._keep_top_per_class(detections)
            results.append(detections)
        return results
    def _keep_top_per_class(self, detections):
        """
        从检测结果中保留每个类别置信度最高的框
        :param detections: (num_detections, 6) 每行 [class_id, confidence, x1, y1, x2, y2]
        :return: 每个类别置信度最高的框组成的张量
        """
        if detections.size(0) == 0:
            return detections
        # 只保留非背景的预测 (class_id > 0)
        valid_mask = detections[:, 0] > 0
        valid_detections = detections[valid_mask]
        if valid_detections.size(0) == 0:
            return torch.zeros(0, 6, device=detections.device)
        # 获取所有类别
        class_ids = valid_detections[:, 0].unique()
        top_detections = []
        for class_id in class_ids:
            # 获取当前类别的所有预测
            class_mask = valid_detections[:, 0] == class_id
            class_detections = valid_detections[class_mask]
            # 找到置信度最高的那个
            best_idx = torch.argmax(class_detections[:, 1])
            top_detections.append(class_detections[best_idx])
        if len(top_detections) == 0:
            return torch.zeros(0, 6, device=detections.device)

        return torch.stack(top_detections)

    @torch.no_grad()
    def test(self, test_dataloader, num_images=5, nms_threshold=0.5,
             pos_threshold=0.5, save_result=True, output_dir='test_results',
              load_path=None):
        """
        测试模型并可视化结果，计算准确率
        Args:
            test_dataloader: 测试数据加载器
            num_images: 要可视化的图片数量
            nms_threshold: NMS阈值
            pos_threshold: 置信度阈值
            save_result: 是否保存结果
            output_dir: 输出目录
        """
        # 如果有预训练模型，加载
        if load_path is not None:
            self.load_model(load_path)
        self.model.eval()
        device = self.device
        print('\nTesting model...')
        # 创建输出目录
        if save_result:
            os.makedirs(output_dir, exist_ok=True)
        # 收集所有预测结果和真实标签
        all_predictions = []# 每张图片的预测结果
        all_labels = []     # 每张图片的真实标签
        all_images = []     # 每张图片
        # 遍历整个测试集
        print('Predicting all images...')
        with tqdm(test_dataloader, desc='Predicting') as pbar:
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(device)
                labels = labels.to(device)
                #  预测 (每个类别只保留置信度最高的框)
                results = self.predict(images, nms_threshold, pos_threshold, keep_top_per_class=True)
                # 存储结果
                for i in range(len(images)):
                    # 确保 predictions 和 labels 都是正确格式
                    pred = results[i]
                    if pred.dim() == 1 and pred.numel() == 6:
                        pred = pred.unsqueeze(0)  # (1, 6)
                    elif pred.dim() == 1 and pred.numel() == 0:
                        pred = torch.zeros(0, 6)
                    elif pred.dim() == 0:
                        pred = torch.zeros(0, 6)
                    label = labels[i]
                    if label.dim() == 1 and label.numel() == 5:
                        label = label.unsqueeze(0)  # (1, 5)
                    elif label.dim() == 1 and label.numel() == 0:
                        label = torch.zeros(0, 5)
                    elif label.dim() == 0:
                        label = torch.zeros(0, 5)

                    all_predictions.append(pred.cpu())
                    all_labels.append(label.cpu())
                    all_images.append(images[i].cpu())

                # 更新进度条
                pbar.set_postfix({'batch': f'{batch_idx + 1}/{len(test_dataloader)}'})
        # ============ 计算正确率 ============
        accuracy_metrics = self.calculate_accuracy(all_predictions, all_labels)
        print('Test Results:')
        print('=' * 50)
        print(f'Total Images: {accuracy_metrics["total_images"]}')
        print(f'Total Ground Truth Boxes: {accuracy_metrics["total_gt_boxes"]}')
        print(f'Total Predicted Boxes: {accuracy_metrics["total_pred_boxes"]}')
        print(f'True Positives: {accuracy_metrics["true_positives"]}')
        print(f'False Positives: {accuracy_metrics["false_positives"]}')
        print(f'False Negatives: {accuracy_metrics["false_negatives"]}')
        print(f'Precision: {accuracy_metrics["precision"]:.4f}')
        print(f'Recall: {accuracy_metrics["recall"]:.4f}')
        print(f'F1 Score: {accuracy_metrics["f1_score"]:.4f}')
        print(f'mAP: {accuracy_metrics["map"]:.4f}')
        # ============ 保存结果 ============
        if save_result:
            # 保存准确率结果
            result_file = os.path.join(output_dir, 'test_results.txt')
            with open(result_file, 'w') as f:
                f.write('Test Results:\n')
                for key, value in accuracy_metrics.items():
                    if isinstance(value, int):
                        f.write(f'{key}: {value}\n')
                    elif isinstance(value, float):
                        f.write(f'{key}: {value:.4f}\n')
                    elif isinstance(value, dict):
                        f.write(f'{key}: {value}\n')  # 也可以保存字典
            print(f'\nResults saved to {result_file}')
        # ============ 可视化结果 ============
        self.visualize_predictions(all_images, all_predictions, all_labels,
                                   num_images=num_images, output_dir=output_dir)
        return accuracy_metrics
    def calculate_accuracy(self, predictions, labels, iou_threshold=0.5):
        """
        计算准确率指标
        Args:
            predictions: 预测结果列表，每个元素是 (num_detections, 6)，每行 [class_id, confidence, x1, y1, x2, y2]
            labels: 真实标签列表，每个元素是 (num_detections, 5)，每行 [class_id, x1, y1, x2, y2]
            iou_threshold: IoU阈值，用于判断是否为正样本

        Returns:
            accuracy_metrics: 包含各种指标的字典
        """
        total_images = len(predictions)
        total_gt_boxes = 0
        total_pred_boxes = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        # 按类别统计
        class_stats = {}
        for pred, label in zip(predictions, labels):
            # ============ 处理真实框 ============
            # label 中 0 是香蕉，1 是其他物体
            # 所有行都是有效类别（没有背景）
            # 确保 gt_boxes 是二维的
            if label.dim() == 1:
                # 如果是一维，可能是 (5,) 即 [class_id, x1, y1, x2, y2]
                # 需要 reshape 成 (1, 5)
                if label.numel() == 5:
                    gt_boxes = label.unsqueeze(0)  # (1, 5)
                elif label.numel() == 0:
                    gt_boxes = torch.zeros(0, 5, dtype=label.dtype)
                else:
                    # 尝试 reshape
                    gt_boxes = label.reshape(-1, 5)
            elif label.dim() == 2:
                gt_boxes = label
            else:
                raise ValueError(f"Unexpected label dimension: {label.dim()}, shape: {label.shape}")
            num_gt = len(gt_boxes)
            total_gt_boxes += num_gt
            if num_gt > 0:
                # 将原始类别+1，使其与预测格式对齐
                gt_classes = gt_boxes[:, 0] + 1  # 0->1(香蕉), 1->2(其他)
                gt_coords = gt_boxes[:, 1:5]  # (num_gt, 4)
            else:
                gt_classes = torch.tensor([], dtype=torch.long)
                gt_coords = torch.tensor([])
            # ============ 处理预测框 ============
            # pred 中 class_id=0 是背景（已被NMS过滤）
            # class_id=1 是香蕉，class_id=2 是其他
            pred_boxes = pred[pred[:, 0] > 0]  # 排除背景
            num_pred = len(pred_boxes)
            total_pred_boxes += num_pred
            # 提取预测信息
            if num_pred > 0:
                pred_classes = pred_boxes[:, 0].long()  # (num_pred,)
                pred_confs = pred_boxes[:, 1]  # (num_pred,)
                pred_coords = pred_boxes[:, 2:6]  # (num_pred, 4)
            else:
                pred_classes = torch.tensor([], dtype=torch.long)
                pred_coords = torch.tensor([])
                pred_confs = torch.tensor([])
            # ============ 处理各种情况 ============
            # ============ 情况1: 没有真实框 ============
            if num_gt == 0:
                # 没有真实框，所有预测都是假阳性
                false_positives += num_pred
                # 按类别统计
                for cls_id in pred_classes.unique():
                    if cls_id.item() not in class_stats:
                        class_stats[cls_id.item()] = {'tp': 0, 'fp': 0, 'fn': 0}
                    count = (pred_classes == cls_id).sum().item()
                    class_stats[cls_id.item()]['fp'] += count
                continue
            # ============ 情况2: 没有预测框 ============
            if num_pred == 0:
                # 没有预测框，所有真实框都是假阴性
                false_negatives += num_gt
                for cls_id in gt_classes.unique():
                    if cls_id.item() not in class_stats:
                        class_stats[cls_id.item()] = {'tp': 0, 'fp': 0, 'fn': 0}
                    count = (gt_classes == cls_id).sum().item()
                    class_stats[cls_id.item()]['fn'] += count
                continue
            # ============ 情况3: 有真实框也有预测框 ============
            # 由于每个类别最多只有1个预测框，直接为每个预测框找匹配的GT
            matched_gt = set()
            # 遍历每个预测框 (每个类别最多1个，所以数量很少)
            for pred_idx in range(num_pred):
                pred_class = pred_classes[pred_idx].item()
                pred_coord = pred_coords[pred_idx]  # (4,)
                pred_conf = pred_confs[pred_idx].item()
                # 找相同类别且IoU最高的GT框
                best_iou = 0
                best_gt_idx = -1
                for gt_idx in range(num_gt):
                    if gt_idx in matched_gt:
                        continue
                    if gt_classes[gt_idx].item() != pred_class:
                        continue
                    # 计算IoU
                    gt_coord = gt_coords[gt_idx]
                    iou = IOU.box_iou(pred_coord.unsqueeze(0), gt_coord.unsqueeze(0))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                # 判断是否匹配成功
                if best_gt_idx >= 0 and best_iou >= iou_threshold:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                    if pred_class not in class_stats:
                        class_stats[pred_class] = {'tp': 0, 'fp': 0, 'fn': 0}
                    class_stats[pred_class]['tp'] += 1
                else:
                    false_positives += 1
                    if pred_class not in class_stats:
                        class_stats[pred_class] = {'tp': 0, 'fp': 0, 'fn': 0}
                    class_stats[pred_class]['fp'] += 1
            # 未匹配的真实框是假阴性
            for gt_idx in range(num_gt):
                if gt_idx not in matched_gt:
                    false_negatives += 1
                    gt_class = gt_classes[gt_idx].item()
                    if gt_class not in class_stats:
                        class_stats[gt_class] = {'tp': 0, 'fp': 0, 'fn': 0}
                    class_stats[gt_class]['fn'] += 1
        # ============ 计算指标 ============
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # 计算mAP
        ap_list = []
        for cls_id, stats in class_stats.items():
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            if tp + fp > 0 and tp + fn > 0:
                ap = tp / (tp + fp + fn)
                ap_list.append(ap)
        map_score = sum(ap_list) / len(ap_list) if ap_list else 0
        return {
            'total_images': total_images,
            'total_gt_boxes': total_gt_boxes,
            'total_pred_boxes': total_pred_boxes,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'map': map_score,
            'class_stats': class_stats
        }

    def visualize_predictions(self, images, predictions, labels, num_images=5, output_dir='test_results'):
        """
        可视化预测结果
        Args:
            images: 图片列表
            predictions: 预测结果列表
            labels: 真实标签列表
            num_images: 要显示的图片数量
            output_dir: 输出目录
        """
        num_images = min(num_images, len(images))
        fig, axes = plt.subplots(num_images, 1, figsize=(10, 6 * num_images))
        if num_images == 1:
            axes = [axes]
        for idx in range(num_images):
            ax = axes[idx]
            # 显示图片
            img = images[idx].permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            ax.imshow(img)
            # 显示真实框 (绿色)
            label = labels[idx]  # (num_gt, 5) [class_id, x1, y1, x2, y2]
            # 确保 label 是二维的
            if label.dim() == 1:
                if label.numel() == 5:
                    label = label.unsqueeze(0)  # (1, 5)
                elif label.numel() == 0:
                    label = torch.zeros(0, 5)
                else:
                    label = label.reshape(-1, 5)
            elif label.dim() == 0:
                label = torch.zeros(0, 5)
            for i in range(label.size(0)):
                x1, y1, x2, y2 = label[i, 1:5].numpy()
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='green', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f'GT_{int(label[i, 0].item()+1)}',
                        color='green', fontsize=8, fontweight='bold')
            # 显示预测框 (红色，每个类别只显示置信度最高的)
            pred = predictions[idx]  # (num_pred, 6) [class_id, confidence, x1, y1, x2, y2]
            for i in range(pred.size(0)):
                if pred[i, 0] > 0:  # 非背景
                    x1, y1, x2, y2 = pred[i, 2:6].numpy()
                    conf = pred[i, 1].item()
                    cls_id = int(pred[i, 0].item())
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, f'Pred_{cls_id}:{conf:.2f}',
                            color='red', fontsize=8, fontweight='bold')
            ax.set_title(f'Image {idx + 1} (Green: GT, Red: Pred)')
            ax.axis('off')
        plt.tight_layout()
        # 保存图片
        save_path = os.path.join(output_dir, 'test_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Visualization saved to {save_path}')

        plt.show()

    def load_model(self, model_path):
        """
        加载预训练模型参数
        Args:
            model_path: 模型权重文件路径
        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False

        try:
            # 加载模型权重
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded successfully from: {model_path}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False