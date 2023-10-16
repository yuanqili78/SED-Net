from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment


class mIoULoss(nn.Module):
    def __init__(self, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target_oneHot, matching_indices=None, gt_mask=None, is_eval=False):
        # inputs => B x Classes x N
        # target_oneHot => B x Classes x N
        # matching_indices  inputs =>  same size
        # gt_mask bool B X C

        B = inputs.shape[0]

        if matching_indices is not None:
            inputs = torch.gather(inputs, dim=1, index=matching_indices)

        # Numerator Product
        inter = inputs * target_oneHot
        # Sum over all pixels B x C x N => B x C
        inter_sum = inter.reshape((B, self.classes, -1)).sum(2)

        # Denominator
        union = inputs + target_oneHot - inter
        # Sum over all pixels B x C x N => B x C
        union = union.reshape((B, self.classes, -1)).sum(2)

        # union[union == 0] = 2e3  # ================================================================
        # print(union.detach().cpu().numpy())

        loss = inter_sum / union  # B X C

        if gt_mask is None:
            return 1 - loss.mean() if not is_eval else (1 - loss.mean(-1)).detach().cpu().numpy()
        else:
            loss = torch.where(gt_mask, loss, torch.zeros_like(loss)).sum(-1)  # B
            loss_mask = gt_mask.int().sum(-1)  # B
            nume = loss.sum()
            denom = torch.nonzero(gt_mask).shape[0]
            return 1 - nume / denom if not is_eval else (1 - loss / loss_mask).detach().cpu().numpy()


class mIoULoss_weight(mIoULoss):
    def __init__(self, n_classes=2, abs_W=False):
        super(mIoULoss_weight, self).__init__()
        self.classes = n_classes
        self.abs_W = abs_W

    def forward(self, inputs, target_oneHot, matching_indices=None, gt_mask=None):
        # inputs => B x Classes x N
        # target_oneHot => B x Classes x N
        # matching_indices inputs =>  same size 
        # gt_mask bool B X C

        B = inputs.shape[0]

        if matching_indices is not None:
            inputs = torch.gather(inputs, dim=1, index=matching_indices)

        # Numerator Product
        inter = inputs * target_oneHot
        # Sum over all pixels B x C x N => B x C
        inter_sum = inter.reshape((B, self.classes, -1)).sum(2)

        # Denominator
        union = inputs + target_oneHot - inter
        # Sum over all pixels B x C x N => B x C
        union = union.reshape((B, self.classes, -1)).sum(2)

        loss = inter_sum / union  # B X C

        
        
        if gt_mask is None:
            target_insts = torch.count_nonzero(target_oneHot.sum(-1), -1).cuda()  # [B, ]  
            target_insts_W = (target_insts / target_insts.sum()).detach()  # [B, ]
            return 1 - (loss.mean(-1) * target_insts_W).sum()
        else:
            target_insts = torch.count_nonzero((target_oneHot.sum(-1)).bool() & gt_mask, -1).cuda()  # [B, ] 

            if not self.abs_W:
                target_insts_W = (target_insts / target_insts.sum()).detach()  # [B, ]
            else:
                target_insts_W = (target_insts / 8).detach()  #   # [B, ]
                target_insts_W = target_insts_W.pow(1.3)
                target_insts_W = target_insts_W / target_insts_W.sum()

            # print(target_insts_W.detach().cpu().numpy())
            loss = torch.where(gt_mask, loss, torch.zeros_like(loss))  # B x C
            return 1 - (loss.mean(-1) * target_insts_W).sum()


def reorder(inputs, target):
    # reorder target  B x N  classes，match the patches
    # target  B x N
    # input B x c x N

    inputs_idx = torch.argmax(inputs.transpose(2, 1), dim=-1).long().to("cpu")  # B x N

    # print("input", inputs_idx)

    B, C, N = inputs.shape[0], inputs.shape[1], inputs.shape[2]

    match_matrix = torch.zeros((B, C, C), dtype=torch.float64)  # gt --> predict 

    for i in range(B):
        for j in range(int(max(target[i])) + 1):  
            primitive_j = torch.nonzero(target[i] == j).flatten()
            # print("primitive_j", primitive_j)
            primitive_j_onehot = torch.zeros((N,), dtype=torch.long)
            primitive_j_onehot[primitive_j] = 1

            # print(primitive_j_onehot)
            inter = inputs_idx[i, primitive_j]  # k

            # print(inter)

            for j_pred in range(C):
                inter_j_pred = torch.nonzero(inter == j_pred).flatten().shape[0]
                if inter_j_pred != 0:
                    # print(j_pred, "has inter", inter_j_pred)
                    primitive_j_pred_onehot = torch.zeros((N,), dtype=torch.long)
                    primitive_j_pred_onehot[torch.nonzero(inputs_idx[i] == j_pred).flatten()] = 1
                    union = (primitive_j_onehot | primitive_j_pred_onehot).sum()
                    # print("union", int(union))
                    match_matrix[i, j, j_pred] = float(inter_j_pred) / float(union)

    match_matrix = match_matrix.numpy()
    # print(match_matrix[0][:8])
    # np.savetxt("./mat.txt", match_matrix.reshape(B * C, C))
    # match
    for i in range(B):
        _, col = linear_sum_assignment(match_matrix[i], True)

        for j in range(int(max(target[i])) + 1):
            target[i, target[i] == j] = col[j]

    return target


def reorder_pred_idx(inputs, target):

    inputs_idx = torch.argmax(inputs.transpose(2, 1), dim=-1).long().to("cpu")  # B x N

    B, C, N = inputs.shape[0], inputs.shape[1], inputs.shape[2]

    target_inst_num = torch.max(target, dim=-1)[0].cpu().numpy() + 1  # [B, ]

    match_matrix = torch.zeros((B, C, C), dtype=torch.float64)  # gt --> predict 

    for i in range(B):
        for j in range(target_inst_num[i]):  
            primitive_j = torch.nonzero(target[i] == j).flatten()
            # print("primitive_j", primitive_j)
            primitive_j_onehot = torch.zeros((N,), dtype=torch.long)
            primitive_j_onehot[primitive_j] = 1

            # print(primitive_j_onehot)
            inter = inputs_idx[i, primitive_j]  # k

            # print(inter)

            for j_pred in range(C):
                inter_j_pred = torch.nonzero(inter == j_pred).flatten().shape[0]
                if inter_j_pred != 0:
                    # print(j_pred, "has inter", inter_j_pred)
                    primitive_j_pred_onehot = torch.zeros((N,), dtype=torch.long)
                    primitive_j_pred_onehot[torch.nonzero(inputs_idx[i] == j_pred).flatten()] = 1
                    union = (primitive_j_onehot | primitive_j_pred_onehot).sum()
                    # print("union", int(union))
                    match_matrix[i, j, j_pred] = float(inter_j_pred) / float(union)

    match_matrix = match_matrix.numpy()
    # match

    matching_indices = np.zeros((B, N, C), dtype=np.long)

    for i in range(B):
        _, col = linear_sum_assignment(match_matrix[i, :target_inst_num[i], :], True)
        matching_indices[i, :, :target_inst_num[i]] = col

    return matching_indices, target_inst_num


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask




class my_mIoU(nn.Module):
    def __init__(self, n_classes=10):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.mIOU = mIoULoss(n_classes=n_classes)
    
    @torch.no_grad()
    def forward(self, predicts, targets):
        """
        predicts: B x 10 x N
        targets: B x N  long
        """
        targets_onehot = torch.functional.one_hot(targets, num_classes=self.classes).float()
        Inst_pred_onehot = torch.functional.one_hot(predicts, num_classes=self.classes).float().transpose(2, 1).cuda()
        matching_indices, target_inst_num = reorder_pred_idx(Inst_pred_onehot, targets)
        



from src.chamfer_distance.chamfer_distance import ChamferIndex
from pointnet2_ops.pointnet2_utils import ThreeNN


def mIoU_Loss_edge(points_gt, inst_pred, edge_cls_pred):
    '''
    inputs: B x classes x N
    edges: B x 1 x N
    '''
    myThreeNN = ThreeNN()   # return nearest 3 points
    inst_pred = inst_pred.max(1)[1] # B x N
    _, myThreeNN = myThreeNN.apply(points_gt.contiguous(), points_gt.contiguous())  # B x N x 3
    myThreeNN = myThreeNN[..., 1]  # B x N
    inst_pred_nearest = torch.gather(inst_pred, dim=-1, index=myThreeNN.long())  # B x N

    inst_edge_pred = (inst_pred_nearest != inst_pred).float()  # B x N
    edge_cls_pred = (edge_cls_pred.max(dim=1)[1] == 1).float()  # B x N

    # print(inst_edge_pred.sum(-1), edge_cls_pred.sum(-1))
    inter = (inst_edge_pred * edge_cls_pred).sum(-1)
    union = inst_edge_pred.sum(-1) + edge_cls_pred.sum(-1) - inter + 1e-7
    return 1 - (inter / union).mean()


if __name__ == "__main__":
    a = sequence_mask(torch.range(4, 8), maxlen=10)
    print(a)






