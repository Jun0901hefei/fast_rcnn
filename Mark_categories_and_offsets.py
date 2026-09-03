import torch
import corner_and_center
import assign_anchor_box
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """
    计算每个锚框与其对应的标准框的偏移量

    假设一共有n个锚框，其中分配到标准框的有m个，那么anchors和assigned_bb的形状都为（m，4）

    :param anchors: 锚框的左上右下坐标
    :param assigned_bb:每个锚框匹配的真实边界框的左上右下坐标
    :param eps:一个很小的数，防止除以零或取对数时出现数值不稳定
    :return:每个锚框相对于其匹配的真实框的偏移量。
    """
    #将锚框从角点格式转换为中心格式
    c_anc = corner_and_center.box_corner_to_center(anchors)
    # 将标准框从角点格式转换为中心格式
    c_assigned_bb = corner_and_center.box_corner_to_center(assigned_bb)
    #计算中心坐标的偏移量
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    #计算高宽的偏移量
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    #合并成[x的偏移量，y的偏移量，w的偏移量，h的偏移量]
    offset = torch.cat([offset_xy, offset_wh], dim=1)
    return offset
def multi_box_target(proposals_batch, labels):
    """
        Fast R-CNN 的候选区域目标生成
        :Arg：
            proposals_batch:
                (batch_size, num_proposals, 4)
                每张图片的 Selective Search 候选区域
            labels:
                (batch_size, num_gt_boxes, 5)
                [class_id, x1, y1, x2, y2]
        :return：
            bbox_offset:
                (batch_size, num_proposals * 4)

            bbox_mask:
                (batch_size, num_proposals * 4)
            class_labels:
                (batch_size, num_proposals)
                0 = background
                >0 = object class
        """

    batch_size = labels.shape[0]
    device = proposals_batch.device
    num_proposals = proposals_batch.shape[1]

    batch_offset = []
    batch_mask = []
    batch_class_labels = []

    for i in range(batch_size):
        # 当前图片的 proposals
        proposals = proposals_batch[i]      # (N, 4)
        # 当前图片的 GT
        label = labels[i]                   # (M, 5)

        # 1. 判断哪些 proposal 是真正的 proposal
        #    padding 是 [0,0,0,0]
        valid_mask = ((proposals[:, 2] > proposals[:, 0]) &(proposals[:, 3] > proposals[:, 1]))
        if not torch.is_tensor(valid_mask):
            valid_mask = torch.tensor(valid_mask, device=proposals.device)
        #所有不是padding的坐标
        valid_indices = torch.nonzero(valid_mask , as_tuple=False).squeeze(1)#(valid_proposals,)

        # 2. 初始化全部 proposal 的目标
        class_labels = torch.zeros(num_proposals,dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_proposals, 4),dtype=torch.float32, device=device)
        bbox_mask = torch.zeros((num_proposals, 4),dtype=torch.float32,device=device)

        # 3. 只有真正的 proposal 才参与 IoU 匹配
        if valid_indices.numel() > 0:
            #所有正样本的提议区域
            valid_proposals = proposals[valid_indices]
            #valid_proposals,)
            anchors_bbox_map = assign_anchor_box.assign_anchor_to_bbox(label[:, 1:],valid_proposals,device)
            # 4. 找到正样本
            positive_mask = anchors_bbox_map >= 0
            positive_indices = torch.nonzero(positive_mask,as_tuple=False).squeeze(1)#（pos_proposals,）
            if positive_indices.numel() > 0:
                # 分配到的bbox在label里面的索引
                bb_idx = anchors_bbox_map[positive_indices]
                # 映射回原 proposals 的索引
                original_indices = valid_indices[positive_indices]
                # 5. 类别标签
                #    原始：banana = 0
                #    Fast R-CNN：background = 0
                class_labels[original_indices] = (label[bb_idx, 0].long() + 1)
                # 6. 真实 bbox
                assigned_bb[original_indices] = label[bb_idx, 1:5]
                # 7. bbox mask
                bbox_mask[original_indices] = 1.0
                # 8. 只对正样本计算 offset
                positive_proposals = proposals[original_indices]
                positive_gt_boxes = assigned_bb[original_indices]
                positive_offsets = offset_boxes(positive_proposals,positive_gt_boxes)
                # 把 offset 放回对应位置
                bbox_offset_temp = torch.zeros((num_proposals, 4),dtype=torch.float32,device=device)
                bbox_offset_temp[ original_indices ] = positive_offsets
            else:
                bbox_offset_temp = torch.zeros((num_proposals, 4),dtype=torch.float32,device=device)
        else:
            bbox_offset_temp = torch.zeros((num_proposals, 4),dtype=torch.float32,device=device)
        # 9. 保存当前图片结果
        batch_offset.append(bbox_offset_temp.reshape(-1))
        batch_mask.append( bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)

    # 10. 合并 batch
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)

    return bbox_offset, bbox_mask, class_labels
