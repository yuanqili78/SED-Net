import torch
from torch import nn
# from chamfer_distance.chamfer_distance import ChamferDistance

import numpy as np

import torch.nn.functional as F

import random

# chamfer_distance = ChamferDistance()


def edge_cls_loss(edges_pred, edges_label, bce_W):
    """
    Bce loss
    edges_pred: B x 2 x N , without softmax
    edges_label: B x N
    bce_W: B x N
    """
    BCEloss = nn.CrossEntropyLoss(reduction='none')  # ---》  softmax

    _loss = (BCEloss(edges_pred, edges_label) * bce_W).mean(-1)
    _loss[bce_W.sum(-1) == 0] = 0  # ignore useless loss
    return torch.mean(_loss)



def compute_embedding_loss(pred_feat, gt_label, t_pull=0.5, t_push=1.5):
    '''
    pred_feat: (B, N, K)
    gt_label: (B, N)
    '''
    batch_size, num_pts, feat_dim = pred_feat.shape
    device = pred_feat.device
    pull_loss = torch.Tensor([0.0]).to(device)
    push_loss = torch.Tensor([0.0]).to(device)
    for i in range(batch_size):
        num_class = gt_label[i].max() + 2

        embeddings = []

        for j in range(num_class):
            mask = (gt_label[i] == (j - 1))
            feature = pred_feat[i][mask]
            if len(feature) == 0:
                continue
            embeddings.append(feature)  # (M, K)

        centers = []

        for feature in embeddings:
            center = torch.mean(feature, dim=0).view(1, -1)
            centers.append(center)

        # intra-embedding loss
        pull_loss_tp = torch.Tensor([0.0]).to(device)
        for feature, center in zip(embeddings, centers):
            dis = torch.norm(feature - center, 2, dim=1) - t_pull
            dis = F.relu(dis)
            pull_loss_tp += torch.mean(dis)

        pull_loss = pull_loss + pull_loss_tp / len(embeddings)

        # inter-embedding loss
        centers = torch.cat(centers, dim=0)  # (num_class, K)

        if centers.shape[0] == 1:
            continue

        dst = torch.norm(centers[:, None, :] - centers[None, :, :], 2, dim=2)

        eye = torch.eye(centers.shape[0]).to(device)
        pair_distance = torch.masked_select(dst, eye == 0)

        pair_distance = t_push - pair_distance
        pair_distance = F.relu(pair_distance)
        push_loss += torch.mean(pair_distance)

    pull_loss = pull_loss / batch_size
    push_loss = push_loss / batch_size
    loss = pull_loss + push_loss
    # print("tripet loss", loss.shape)
    return loss, pull_loss, push_loss


my_nllLoss = torch.nn.NLLLoss()

def compute_edge_embedding_loss(edges_pred, pred_feat, gt_label, edges_num=2000, use_type=False, primitives=None, primitives_log_prob=None):

    edges_idx = torch.argsort(edges_pred[:, 1, :], dim=-1, descending=True)  # B x N

    pred_feat = pred_feat.transpose(1, 2)  # B x N x K

    pred_feat = torch.gather(pred_feat, dim=1, index=edges_idx.unsqueeze(-1).expand(edges_idx.shape[0], edges_idx.shape[1], pred_feat.shape[-1]))[:, :edges_num, :].contiguous()  # B x 2000 x K
    gt_label = torch.gather(gt_label, dim=1, index=edges_idx)[:, :edges_num].contiguous()  # B x 2000 

    if not use_type:
        return torch.mean(compute_embedding_loss(pred_feat, gt_label)[0])
    else:
        primitives_log_prob = torch.gather(primitives_log_prob.transpose(1, 2), dim=1, index=edges_idx.unsqueeze(-1).expand(edges_idx.shape[0], edges_idx.shape[1], 6))[:, :edges_num, :].contiguous() 
        primitives = torch.gather(primitives, dim=1, index=edges_idx)[:, :edges_num].contiguous()
        #print(primitives_log_prob.transpose(1, 2).shape, primitives.shape)
        edge_bce_loss = my_nllLoss(primitives_log_prob.transpose(1, 2), primitives)
        return edge_bce_loss + torch.mean(compute_embedding_loss(pred_feat, gt_label)[0])


if __name__ == "__main__":
    a = torch.ones((1,2, 3)).cuda()
    b = torch.zeros((1,1, 3)).cuda()
    print(chamfer_distance(a, b))