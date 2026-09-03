import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.ops as ops
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple
import anchor_box_generator
import assign_anchor_box
import corner_and_center
import data_precession
import fast_rcnn
import fast_rcnn_train
import IOU
import loss_picture
import Mark_categories_and_offsets
import NMSandMulti_box_detection
import predict
import show_boundingbox

batch_size = 20
train_iter, test_iter = data_precession.load_data_bananas(batch_size)
# 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = fast_rcnn.FastRCNN(
        num_classes=2,  # 香蕉 + 背景
        roi_output_size=(7, 7),
        ss_mode='fast',
        max_proposals=1000
    )
model = model.to(device)
# 创建损失历史记录
loss_history = loss_picture.LossHistory(save_dir='loss_history')

# 创建训练器
trainer = fast_rcnn_train.FastRCNNTrainer(model, device=device, loss_history=loss_history)

# 训练
trainer.train(train_iter, num_epochs=10, learning_rate=0.00001, save_dir='checkpoints')

# 测试
trainer.test(test_iter, num_images=5, nms_threshold=0.5, pos_threshold=0.5)


