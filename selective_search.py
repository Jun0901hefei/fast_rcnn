import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as patches
class SelectiveSearch:
    """
    Selective Search 候选区域生成器
    使用 OpenCV 实现
    """
    def __init__(self, mode='fast', max_rois=2000, min_size=20):
        """
        初始化 Selective Search
        Args:
            mode: 'fast' 或 'quality'
            max_rois: 最大候选区域数量
            min_size: 最小区域大小
        """
        self.mode = mode
        self.max_rois = max_rois
        self.min_size = min_size
    def generate_proposals(self, image):
        """
        生成候选区域
        Args:
            image: (H, W, 3) RGB 图像，值域 [0, 1] 或 [0, 255]
        Returns:
            proposals: (N, 4) 候选区域 [x1, y1, x2, y2]
            scores: (N,) 每个区域的分数
        """
        # 转换为 OpenCV 格式
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        elif image.dtype == np.uint8:
            image = image.copy()
        else:
            image = image.astype(np.uint8)
        # RGB -> BGR (OpenCV 使用 BGR)
        if image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        # 创建 Selective Search 分割对象
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image_bgr)
        # 设置模式
        if self.mode == 'fast':
            ss.switchToSelectiveSearchFast()
        else:
            ss.switchToSelectiveSearchQuality()
        # 生成候选区域
        rects = ss.process()
        # 过滤和限制数量
        proposals = []
        scores = []
        for rect in rects:
            x, y, w, h = rect
            # 过滤太小的区域
            if w < self.min_size or h < self.min_size:
                continue
            # 过滤太细长的区域
            aspect_ratio = w / h
            if aspect_ratio > 3 or aspect_ratio < 0.33:
                continue
            # 转换为 [x1, y1, x2, y2]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            proposals.append([x1, y1, x2, y2])
            # 简单分数（面积越大分数越高）
            scores.append(w * h)
        proposals = np.array(proposals)
        scores = np.array(scores)

        if len(proposals) > 0:
            # 按分数降序排序
            sorted_indices = np.argsort(scores)[::-1]
            proposals = proposals[sorted_indices]
            scores = scores[sorted_indices]
        # 限制数量
        if len(proposals) > self.max_rois:
            proposals = proposals[:self.max_rois]
            scores = scores[:self.max_rois]
        return proposals, scores