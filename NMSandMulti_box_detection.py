import numpy as np
import torch
import IOU
import predict
def nms(boxes, scores, iou_threshold):
    """
    输出的是置信度最高的框和与之不重叠的框
    :param boxes:所有预测边界框的坐标(左上右下)
    :param scores:每个预测框对应的置信度
    :param iou_threshold:如果两个框的 IoU 大于该阈值，则认为它们过于相似，需要抑制其中置信度较低的
    :return:保留下来的预测框在原始 boxes 中的索引列表
    """
    # 没有候选框
    if boxes.numel() == 0:
        return torch.empty( 0, dtype=torch.long, device=boxes.device )
    #按置信度从高到低排列的框的索引
    order = torch.argsort( scores, descending=True )
    #保留预测边界框的指标
    keep = []
    while order.numel() > 0:
        #取出当前置信度最高的框
        current_index = order[0]
        # 保存原始索引
        keep.append(current_index)
        # 只剩一个框
        if order.numel() == 1:
            break
        # 当前框
        current_box = boxes[current_index].reshape(1, 4)
        # 剩余框的索引
        remaining_indices = order[1:]
        # 剩余框
        remaining_boxes = boxes[remaining_indices]
        # 计算 IoU
        ious = IOU.box_iou(current_box, remaining_boxes).reshape(-1)
        # 保留 IoU <= threshold 的框
        keep_mask = ious <= iou_threshold
        order = remaining_indices[keep_mask]
    return torch.stack(keep)
def multi_box_detection(cls_probs, offset_preds, proposals, nms_threshold=0.5,
                        pos_threshold=0.009999999):
    """
    :param cls_probs:(batch_size, num_classes, num_proposals)每个锚框属于每个类别的概率
    :param offset_preds:(batch_size, num_proposals * 4)每个锚框的4个偏移量预测
    :param proposals:(batch_size, num_proposals, 4)所有锚框的坐标
    :param nms_threshold:NMS 的 IoU 阈值，默认 0.5
    :param pos_threshold:置信度阈值，低于此值视为背景，默认 0.009999999
    :return:输出每个锚框的[类别, 置信度, x1, y1, x2, y2]
    """
    device = cls_probs.device
    batch_size = cls_probs.shape[0]
    num_classes = cls_probs.shape[1]
    num_proposals = cls_probs.shape[2]
    # 检查 proposals
    if proposals.dim() != 3:
        raise ValueError("proposals 必须是 [batch_size, num_proposals, 4]")
    if proposals.shape[0] != batch_size:
        raise ValueError("proposals 的 batch_size 与 cls_probs 不一致")
    if proposals.shape[1] != num_proposals:
        raise ValueError("proposals 数量与 cls_probs 中的 num_proposals 不一致")
    if proposals.shape[2] != 4:
        raise ValueError("proposals 最后一维必须为 4")
    # 保存 batch 输出
    outputs = []
    for batch_idx in range(batch_size):
        # 当前图片的 proposals
        proposals_i = proposals[batch_idx]# [N, 4]
        # 当前图片分类概率
        cls_prob_i = cls_probs[batch_idx]# [C, N]
        # 当前图片 bbox regression
        offset_pred_i = offset_preds[batch_idx].reshape(num_proposals, 4)
        foreground_prob = cls_prob_i[1:]#现在又变成了0是物体类
        conf, class_id = torch.max(foreground_prob,dim=0)
        class_id = class_id + 1
        # 2. bbox offset 解码
        predicted_boxes = predict.offset_inverse(proposals_i,offset_pred_i)
        # 3. NMS
        keep = nms(predicted_boxes,conf,nms_threshold)
        # 4. 默认全部设置为 background
        final_class_id = torch.zeros(num_proposals,dtype=torch.long,device=device)
        final_conf = torch.zeros(num_proposals,dtype=conf.dtype,device=device)
        # 5. 保留 NMS 后的框
        final_class_id[keep] = class_id[keep]
        final_conf[keep] = conf[keep]
        # 6. confidence threshold
        below_threshold = final_conf < pos_threshold
        # 低于阈值的设置成 background
        final_class_id[below_threshold] = 0
        # background confidence
        final_conf[below_threshold] = (1.0 - final_conf[below_threshold])
        # 7. 组织最终结果
        pred_info_i = torch.cat([final_class_id.float().unsqueeze(1),
                                 final_conf.unsqueeze(1),
                                 predicted_boxes],dim=1)
        outputs.append(pred_info_i)

    outputs = torch.stack(outputs,dim=0)

    return outputs