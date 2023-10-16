"""
This script defines loss functions for AE based training.
"""
import numpy as np
import torch
from torch.nn import ReLU

from src.mean_shift import MeanShift

meanshift = MeanShift()
WEIGHT = False
relu = ReLU()
'''
if WEIGHT:
    nllloss = torch.nn.NLLLoss(weight=old_weight)
else:'''

nllloss = torch.nn.NLLLoss()


class EmbeddingLoss:
    def __init__(self, margin=1.0, if_mean_shift=False):
        """
        Defines loss function to train embedding network.
        :param margin: margin to be used in triplet loss.
        :param if_mean_shift: bool, whether to use mean shift
        iterations. This is only used in end to end training.
        """
        self.margin = margin
        self.if_mean_shift = if_mean_shift
        self.triplet_loss_nn = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    def triplet_loss(self, output, labels: np.ndarray, iterations=5):
        """
        Triplet loss
        :param output: output embedding from the network. size: B x 128 x N
        where B is the batch size, 128 is the dim size and N is the number of points.
        :param labels: B x N
        """
        max_segments = 5
        batch_size = output.shape[0]
        N = output.shape[2]
        loss_diff = torch.tensor([0.], requires_grad=True).cuda()
        relu = torch.nn.ReLU()

        output = output.permute(0, 2, 1)  # B x N x 128
        output = torch.nn.functional.normalize(output, p=2, dim=2)
        new_output = []

        if self.if_mean_shift:
            for b in range(batch_size):
                new_X, bw = meanshift.mean_shift(output[b], 4000,
                                                 0.015, iterations=iterations,
                                                 nms=False)
                new_output.append(new_X)
            output = torch.stack(new_output, 0)

        num_sample_points = {}
        sampled_points = {}
        for i in range(batch_size):
            sampled_points[i] = {}
            p = labels[i]  # N
            unique_labels = np.unique(p)

            # number of points from each cluster.
            num_sample_points[i] = min([N // unique_labels.shape[0] + 1, 30])
            for l in unique_labels:
                ix = np.isin(p, l)
                sampled_indices = np.where(ix)[0]
                # point indices that belong to a certain cluster.
                sampled_points[i][l] = np.random.choice(
                    list(sampled_indices),
                    num_sample_points[i],
                    replace=True)

        sampled_predictions = {}
        for i in range(batch_size):
            sampled_predictions[i] = {}
            for k, v in sampled_points[i].items():
                pred = output[i, v, :]  # 128 embedding
                sampled_predictions[i][k] = pred

        all_satisfied = 0
        only_one_segments = 0
        for i in range(batch_size):
            len_keys = len(sampled_predictions[i].keys())
            keys = list(sorted(sampled_predictions[i].keys()))
            num_iterations = min([max_segments * max_segments, len_keys * len_keys])
            normalization = 0
            if len_keys == 1:
                only_one_segments += 1
                continue

            loss_shape = torch.tensor([0.], requires_grad=True).cuda()
            for _ in range(num_iterations):
                k1 = np.random.choice(len_keys, 1)[0]
                k2 = np.random.choice(len_keys, 1)[0]
                if k1 == k2:
                    continue
                else:
                    normalization += 1

                pred1 = sampled_predictions[i][keys[k1]]
                pred2 = sampled_predictions[i][keys[k2]]

                Anchor = pred1.unsqueeze(1)
                Pos = pred1.unsqueeze(0)
                Neg = pred2.unsqueeze(0)

                diff_pos = torch.sum(torch.pow((Anchor - Pos), 2), 2)
                diff_neg = torch.sum(torch.pow((Anchor - Neg), 2), 2)
                constraint = diff_pos - diff_neg + self.margin
                constraint = relu(constraint)

                # remove diagonals corresponding to same points in anchors
                loss = torch.sum(constraint) - constraint.trace()

                satisfied = torch.sum(constraint > 0) + 1.0
                satisfied = satisfied.type(torch.cuda.FloatTensor)

                loss_shape = loss_shape + loss / satisfied.detach()

            loss_shape = loss_shape / (normalization + 1e-8)
            loss_diff = loss_diff + loss_shape
        loss_diff = loss_diff / (batch_size - only_one_segments + 1e-8)
        return loss_diff


    def myTripletMarginLoss(self, output, labels: np.ndarray, iterations=5):
        output = output.permute(0, 2, 1)  # B x N x 128
        output = torch.nn.functional.normalize(output, p=2, dim=2)


def evaluate_miou(gt_labels, pred_labels):
    N = gt_labels.shape[0]
    C = pred_labels.shape[2]
    pred_labels = np.argmax(pred_labels, 2)
    IoU_category = 0

    for n in range(N):
        label_gt = gt_labels[n]
        label_pred = pred_labels[n]
        IoU_part = 0.0

        for label_idx in range(C):
            locations_gt = (label_gt == label_idx)
            locations_pred = (label_pred == label_idx)
            I_locations = np.logical_and(locations_gt, locations_pred)
            U_locations = np.logical_or(locations_gt, locations_pred)
            I = np.sum(I_locations) + np.finfo(np.float32).eps
            U = np.sum(U_locations) + np.finfo(np.float32).eps
            IoU_part = IoU_part + I / U
        IoU_sample = IoU_part / C
        IoU_category += IoU_sample
    return IoU_category / N


""" 
#PyTorch
class my_miou_loss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(my_miou_loss, self).__init__()

    def forward(self, inputs, targets, smooth=1.):
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=targets.shape[1]).transpose(1, 2)  # B x K x N 
        B = targets.shape[0]
        
        intersection = (inputs * targets).sum(dim=1) # B x N

        per_face_dice_average = 0.
        sum = 0
        inputs_max = torch.argmax(inputs, dim=1) # B x N
        for b in range(B):
            for i in torch.unique(targets[b]):
                sum+=1
                inter = (intersection[b])[targets[b]==i].sum()
                sum1 = (inputs_max[b])[inputs_max[b] == i]
                sum2 = (inputs_max[b])[inputs_max[b] == i]
                per_face_dice_average += (2.*inter + smooth)/(sum1 + sum2 + smooth)
        
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - per_face_dice_average / sum """




def my_miou_loss(targets, inputs, smooth=1.):
    """
    B x N 
    B x N x C
    """
    targets = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1]).transpose(1, 2) 

    targets, inputs = targets.flatten(), inputs.flatten()

    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

    return 1. - dice

def primitive_loss(pred, gt):
    return nllloss(pred, gt)



class LabelSmoothingLoss(torch.nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.2):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, logprobs, target):
        # logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()



