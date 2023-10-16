"""
This file contains utility functions to segment the embedding using clustering algorithms.
"""
import numpy as np

random_state = 170
from sklearn.cluster import SpectralClustering, KMeans, MeanShift, estimate_bandwidth
import torch
from lapsolver import solve_dense
from src.utils import sample_mesh, triangle_area_multi
from src.utils import chamfer_distance


def cluster(X, number_cluster, bandwidth=None, alg="kmeans"):
    X = X.astype(np.float32)
    if alg == "kmeans":
        y_pred = KMeans(n_clusters=number_cluster, random_state=random_state).fit_predict(X)

    elif alg == "spectral":
        y_pred = SpectralClustering(n_clusters=number_cluster, random_state=random_state, n_jobs=10).fit_predict(X)

    elif alg == "meanshift":
        # There is a little insight here, the number of neighbors are somewhat
        # dependent on the number of neighbors used in the dynamic graph network.
        if bandwidth:
            pass
        else:
            bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=1000)
        seeds = X[np.random.choice(np.arange(X.shape[0]), 5000)]
        # y_pred = MeanShift(bandwidth=bandwidth).fit_predict(X)
        clustering = MeanShift(bandwidth=bandwidth, seeds=seeds, n_jobs=32).fit(X)
        y_pred = clustering.predict(X)

    if alg == "meanshift":
        return y_pred, clustering.cluster_centers_, bandwidth
    else:
        return y_pred


def cluster_prob(embedding, centers):
    """
    Returns cluster probabilities.
    :param embedding: N x 128, embedding for each point
    :param centers: C x 128, embedding for centers
    """
    # should of size N x C
    dot_p = np.dot(centers, embedding.transpose()).transpose()

    prob = np.exp(dot_p) / np.expand_dims(np.sum(np.exp(dot_p), 1), 1)
    return prob


def cluster_prob(embedding, centers, band_width):
    """
    Returns cluster probabilities.
    :param embedding: N x 128, embedding for each point
    :param centers: C x 128, embedding for centers
    """
    dist = 2 - 2 * centers @ embedding.T
    prob = np.exp(-dist / 2 / (band_width)) / np.sqrt(2 * np.pi * band_width)
    return prob


def cluster_prob_mutual(embedding, centers, bandwidth, if_normalize=False):
    """
    Returns cluster probabilities.
    :param embedding: N x 128, embedding for each point
    :param centers: C x 128, embedding for centers
    """
    # dim: C x N
    dist = np.exp(centers @ embedding.T / bandwidth)
    prob = dist / np.sum(dist, 0, keepdims=True)

    if if_normalize:
        prob = prob - np.min(prob, 1, keepdims=True)
        prob = prob / np.max(prob, 1, keepdims=True)
    return prob


def dot_product_from_cluster_centers(embedding, centers):
    return centers @ embedding.T


def sample_from_collection_of_mesh(Meshes, N=10000):
    A = []
    sampled_points = []
    new_meshes = []
    for mesh in Meshes:
        new_mesh = mesh.remove_unreferenced_vertices()
        if np.array(new_mesh.vertices).shape[0] == 0:
            continue
        else:
            new_meshes.append(new_mesh)

    for mesh in new_meshes:
        mesh.remove_unreferenced_vertices()
        vertices = np.array(mesh.vertices)[np.array(mesh.triangles)]
        v1 = vertices[:, 0]
        v2 = vertices[:, 1]
        v3 = vertices[:, 2]

        A.append(np.sum(triangle_area_multi(v1, v2, v3)))

    area = np.sum(A)
    Points = []

    for index, mesh in enumerate(new_meshes):
        if np.array(mesh.vertices).shape[0] == 0:
            continue
        mesh.remove_unreferenced_vertices()
        vertices = np.array(mesh.vertices)[np.array(mesh.triangles)]
        v1 = vertices[:, 0]
        v2 = vertices[:, 1]
        v3 = vertices[:, 2]
        n = int((N * A[index]) // area)
        if n > 10:
            # , face_normals=np.array(mesh.triangle_normals)
            points, normals, _ = sample_mesh(v1, v2, v3, n=n, norms=False)
            try:
                Points.append(points)
            except:
                pass
    Points = np.concatenate(Points, 0)
    return Points.astype(np.float32)


def mean_IOU_one_sample(pred, gt, C):
    IoU_part = 0.0
    for label_idx in range(C):
        locations_gt = (gt == label_idx)
        locations_pred = (pred == label_idx)
        I_locations = np.logical_and(locations_gt, locations_pred)
        U_locations = np.logical_or(locations_gt, locations_pred)
        I = np.sum(I_locations) + np.finfo(np.float32).eps
        U = np.sum(U_locations) + np.finfo(np.float32).eps
        IoU_part = IoU_part + I / U
    return IoU_part / C


def SIOU_matched_segments(target, pred_labels, primitives_pred, primitives, weights):
    """
    Computes iou for segmentation performance and primitive type
    prediction performance.
    First it computes the matching using hungarian matching
    between predicted and ground truth labels.
    Then it computes the iou score, starting from matching pairs
    coming out from hungarian matching solver. Note that
    it is assumed that the iou is only computed over matched pairs.
    That is to say, if any column in the matched pair has zero
    number of points, that pair is not considered.
    
    It also computes the iou for primitive type prediction. In this case
    iou is computed only over the matched segments.
    """
    # 2 is open spline and 9 is close spline
    primitives[primitives == 0] = 9
    primitives[primitives == 6] = 9
    primitives[primitives == 7] = 9
    primitives[primitives == 8] = 2

    primitives_pred[primitives_pred == 0] = 9
    primitives_pred[primitives_pred == 6] = 9
    primitives_pred[primitives_pred == 7] = 9
    primitives_pred[primitives_pred == 8] = 2

    labels_one_hot = to_one_hot(target)
    cluster_ids_one_hot = to_one_hot(pred_labels)

    cost = relaxed_iou_fast(torch.unsqueeze(cluster_ids_one_hot, 0).float(), torch.unsqueeze(labels_one_hot, 0).float())
    cost_ = 1.0 - cost.data.cpu().numpy()
    matching = []

    for b in range(1):
        rids, cids = solve_dense(cost_[b])
        matching.append([rids, cids])

    primitives_pred_hot = to_one_hot(primitives_pred, 10, weights.device.index).float()

    # this gives you what primitive type the predicted segment has.
    prim_pred = primitive_type_segment_torch(primitives_pred_hot, weights).data.cpu().numpy()
    target = np.expand_dims(target, 0)
    pred_labels = np.expand_dims(pred_labels, 0)
    prim_pred = np.expand_dims(prim_pred, 0)
    primitives = np.expand_dims(primitives, 0)

    segment_iou, primitive_iou, iou_b_prims, segment_reacall = mean_IOU_primitive_segment(matching, pred_labels, target, prim_pred,
                                                                         primitives)
    return segment_iou, primitive_iou, matching, iou_b_prims, segment_reacall





def SIOU_matched_segments_usecd(target, pred_labels, primitives_pred, primitives, weights, points):
    """
    Computes iou for segmentation performance and primitive type
    prediction performance.
    First it computes the matching using hungarian matching
    between predicted and ground truth labels.
    Then it computes the iou score, starting from matching pairs
    coming out from hungarian matching solver. Note that
    it is assumed that the iou is only computed over matched pairs.
    That is to say, if any column in the matched pair has zero
    number of points, that pair is not considered.
    
    It also computes the iou for primitive type prediction. In this case
    iou is computed only over the matched segments.
    """
    # 2 is open spline and 9 is close spline
    primitives[primitives == 0] = 9
    primitives[primitives == 6] = 9
    primitives[primitives == 7] = 9
    primitives[primitives == 8] = 2

    primitives_pred[primitives_pred == 0] = 9
    primitives_pred[primitives_pred == 6] = 9
    primitives_pred[primitives_pred == 7] = 9
    primitives_pred[primitives_pred == 8] = 2

    labels_one_hot = to_one_hot(target)
    cluster_ids_one_hot = to_one_hot(pred_labels)

    cost = relaxed_iou_fast(torch.unsqueeze(cluster_ids_one_hot, 0).float(), torch.unsqueeze(labels_one_hot, 0).float())
    cost_ = 1.0 - cost.data.cpu().numpy()
    matching = []

    for b in range(1):
        rids, cids = solve_dense(cost_[b])
        matching.append([rids, cids])

    primitives_pred_hot = to_one_hot(primitives_pred, 10, weights.device.index).float()

    # this gives you what primitive type the predicted segment has.
    prim_pred = primitive_type_segment_torch(primitives_pred_hot, weights).data.cpu().numpy()
    target = np.expand_dims(target, 0)
    pred_labels = np.expand_dims(pred_labels, 0)
    prim_pred = np.expand_dims(prim_pred, 0)
    primitives = np.expand_dims(primitives, 0)

    segment_iou, primitive_iou, iou_b_prims, segment_reacall = mean_IOU_primitive_segment_usecd(matching, pred_labels, target, prim_pred,
                                                                         primitives, points)
    return segment_iou, primitive_iou, matching, iou_b_prims, segment_reacall



import torch.nn as nn
def get_one_hot(targets, nb_classes):
    #res = np.eye(nb_classes)[np.array(targets, dtype=np.int8).reshape(-1)]
    # check none-type
    #if targets.min() == -1:
    #   idx = np.argwhere(targets == -1)
    #   res[idx] = 0
    #return res.reshape(list(targets.shape)+[nb_classes])

    one_hot = nn.functional.one_hot(targets, nb_classes)

    return one_hot
def hungarian_matching(W_pred, W_gt):
    # This non-tf function does not backprob gradient, only output matching indices
    # W_pred - NxK
    # W_gt - NxK'
    # Output: matching_indices
    # The matching does not include gt background instance
    # calculate RIoU
    #n_max_labels = min(W_gt.shape[1], W_pred.shape[1])
    #matching_indices = np.zeros([n_max_labels], dtype=np.int32)

    dot = np.sum(np.expand_dims(W_pred, axis=2) * np.expand_dims(W_gt, axis=1),
                 axis=0)  # K'xK
    denominator = np.expand_dims(np.sum(W_pred, axis=0),
                                 axis=1) + np.expand_dims(np.sum(W_gt, axis=0),
                                                          axis=0) - dot
    cost = dot / np.maximum(denominator, 1e-10)  # K'xK
    row_ind, col_ind = solve_dense(-cost)  # want max solution
    #matching_indices[b, :n_gt_labels] = col_ind

    return row_ind, col_ind
"""     
def compute_riou(W_pred, W_gt, pred_ind, gt_ind):
    # W_pred - NxK
    # W_gt - NxK'

    N, _ = W_pred.shape

    pred_ind = torch.LongTensor(pred_ind).unsqueeze(0).repeat(N, 1).to(
        W_pred.device)
    gt_ind = torch.LongTensor(gt_ind).unsqueeze(0).repeat(N, 1).to(W_gt.device)

    W_pred_reordered = torch.gather(W_pred, -1, pred_ind)
    W_gt_reordered = torch.gather(W_gt, -1, gt_ind)

    dot = torch.sum(W_gt_reordered * W_pred_reordered, dim=0)  # K
    denominator = torch.sum(W_gt_reordered, dim=0) + torch.sum(
        W_pred_reordered, dim=0) - dot
    mIoU = dot / (denominator + 1e-10)  # K
    return mIoU
 """
def npy(var):
    return var.data.cpu().numpy()
def compute_type_miou_abc(type_per_point, T_gt, cluster_pred, I_gt):
    '''
    compute per-primitive-instance type iou
    type_per_point: (1, N, K), K = 6/10
    T_gt: (1, N)
    '''
    assert (type_per_point.shape[0] == 1)
    
    # get T_pred: (1, N)
    if len(type_per_point.shape) == 3:
        B, N, _ = type_per_point.shape
        T_pred = torch.argmax(type_per_point, dim=-1) # (B, N)
    else:
        T_pred = type_per_point

    T_pred[T_pred == 6] = 0
    T_pred[T_pred == 7] = 0
    T_pred[T_pred == 9] = 0
    T_pred[T_pred == 8] = 2
    
    T_gt[T_gt == 6] = 0
    T_gt[T_gt == 7] = 0
    T_gt[T_gt == 9] = 0
    T_gt[T_gt == 8] = 2

    one_hot_pred = get_one_hot(cluster_pred,
                               cluster_pred.max() + 1)[0]  # (N, K)

    if I_gt.min() == -1:
        # (N, K'), remove background
        one_hot_gt = get_one_hot(I_gt + 1,
                                 I_gt.max() + 2)[0][:, 1:]  
    else:
        one_hot_gt = get_one_hot(I_gt, I_gt.max() + 1)[0]

    pred_ind, gt_ind = hungarian_matching(npy(one_hot_pred), npy(one_hot_gt))
    type_iou = torch.Tensor([0.0]).to(T_gt.device)
    cnt = 0

    for p_ind, g_ind in zip(pred_ind, gt_ind):

        try:
            gt_type_label = T_gt[I_gt == g_ind].mode()[0]
            try:
                pred_type_label = T_pred[cluster_pred == p_ind].mode()[0]
            except:
                continue
            if gt_type_label == pred_type_label:
                type_iou += 1
            
            cnt += 1
        except:
            continue
    
    type_iou /= cnt
    return type_iou[0]



def mean_IOU_primitive_segment(matching, predicted_labels, labels, pred_prim, gt_prim):
    """
    Primitive type IOU, this is calculated over the segment level.
    First the predicted segments are matched with ground truth segments,
    then IOU is calculated over these segments.
    :param matching
    :param pred_labels: N x 1, pred label id for segments
    :param gt_labels: N x 1, gt label id for segments
    :param pred_prim: K x 1, pred primitive type for each of the predicted segments
    :param gt_prim: N x 1, gt primitive type for each point
    """
    batch_size = labels.shape[0]
    IOU = []
    RECALL = []
    IOU_prim = []

    for b in range(batch_size):
        iou_b = []
        recall_b = []
        iou_b_prim = []
        iou_b_prims = []
        len_labels = np.unique(predicted_labels[b]).shape[0]
        rows, cols = matching[b]
        count = 0
        for r, c in zip(rows, cols):
            pred_indices = predicted_labels[b] == r
            gt_indices = labels[b] == c

            # use only matched segments for evaluation
            if (np.sum(gt_indices) == 0) or (np.sum(pred_indices) == 0):
                continue

            # also remove the gt labels that are very small in number
            if np.sum(gt_indices) < 100:
                continue
            
            tp = np.sum(np.logical_and(pred_indices, gt_indices))

            iou = tp / (np.sum(np.logical_or(pred_indices, gt_indices)) + 1e-8)
            
            # cal_recall = tp / (tp + fn)
            # https://blog.csdn.net/weixin_39347054/article/details/105408293
            recall = tp / (tp + np.sum(np.logical_and(pred_indices == False, gt_indices == True)) + 1e-8)            

            iou_b.append(iou)
            recall_b.append(recall)
            # evaluation of primitive type prediction performance
            gt_prim_type_k = gt_prim[b][gt_indices][0]
            try:
                predicted_prim_type_k = pred_prim[b][r]
            except:
                print("wrong")

            iou_b_prim.append(gt_prim_type_k == predicted_prim_type_k)
            iou_b_prims.append([gt_prim_type_k, predicted_prim_type_k])

        # find the mean of IOU over this shape
        IOU.append(np.mean(iou_b))
        RECALL.append(np.mean(recall_b))
        IOU_prim.append(np.mean(iou_b_prim))
    return np.mean(IOU), np.mean(IOU_prim), iou_b_prims, np.mean(RECALL)




def mean_IOU_primitive_segment_usecd(matching, predicted_labels, labels, pred_prim, gt_prim, points):
    """
    Primitive type IOU, this is calculated over the segment level.
    First the predicted segments are matched with ground truth segments,
    then IOU is calculated over these segments.
    :param matching
    :param pred_labels: N x 1, pred label id for segments
    :param gt_labels: N x 1, gt label id for segments
    :param pred_prim: K x 1, pred primitive type for each of the predicted segments
    :param gt_prim: N x 1, gt primitive type for each point
    """
    batch_size = labels.shape[0]
    IOU = []
    RECALL = []
    IOU_prim = []

    for b in range(batch_size):
        iou_b = []
        recall_b = []
        iou_b_prim = []
        iou_b_prims = []
        len_labels = np.unique(labels[b]).shape[0]
        rows, cols = matching[b]
        count = 0

        recall_pos = 0

        for r, c in zip(rows, cols):
            pred_indices = predicted_labels[b] == r
            gt_indices = labels[b] == c

            # use only matched segments for evaluation
            if (np.sum(gt_indices) == 0) or (np.sum(pred_indices) == 0):
                continue

            # also remove the gt labels that are very small in number
            # ====================================================================== 小 instance 也考虑
            # if np.sum(gt_indices) < 100:
            #     continue
            
            tp = np.sum(np.logical_and(pred_indices, gt_indices))

            iou = tp / (np.sum(np.logical_or(pred_indices, gt_indices)) + 1e-8)
            
            # cal_recall = tp / (tp + fn)
            # https://blog.csdn.net/weixin_39347054/article/details/105408293
            # recall = tp / (tp + np.sum(np.logical_and(pred_indices == False, gt_indices == True)) + 1e-8)       
            # 

            chamfer_dis = chamfer_distance(points[pred_indices, :].unsqueeze(0), points[gt_indices, :].unsqueeze(0)) / 2
            # print(chamfer_dis)
            if chamfer_dis < 0.1:
                recall_pos += 1

            iou_b.append(iou)
            # recall_b.append(recall)
            # evaluation of primitive type prediction performance
            gt_prim_type_k = gt_prim[b][gt_indices][0]
            try:
                predicted_prim_type_k = pred_prim[b][r]
            except:
                print("wrong")

            iou_b_prim.append(gt_prim_type_k == predicted_prim_type_k)
            iou_b_prims.append([gt_prim_type_k, predicted_prim_type_k])

        # find the mean of IOU over this shape
        IOU.append(np.mean(iou_b))
        RECALL.append(float(recall_pos) / len_labels)
        IOU_prim.append(np.mean(iou_b_prim))
    return np.mean(IOU), np.mean(IOU_prim), iou_b_prims, np.mean(RECALL)



def primitive_type_segment(pred, weights):
    """
    Returns the primitive type for every segment in the predicted shape.
    :param pred: N x L
    :param weights: N x k
    """
    d = np.expand_dims(pred, 2) * np.expand_dims(weights, 1)
    d = np.sum(d, 0)
    return np.argmax(d, 0)


def primitive_type_segment_torch(pred, weights):
    """
    Returns the primitive type for every segment in the predicted shape.
    :param pred: N x L
    :param weights: N x k
    """
    d = torch.unsqueeze(pred, 2) * torch.unsqueeze(weights, 1)
    d = torch.sum(d, 0)
    return torch.max(d, 0)[1]


def iou_segmentation(pred, gt):
    # preprocess the predictions and gt to remove the extras
    # swap (0, 6, 7) to closed surfaces which is 9
    # swap 8 to 2
    gt[gt == 0] = 9
    gt[gt == 6] = 9
    gt[gt == 7] = 9
    gt[gt == 8] = 2

    pred[pred == 0] = 9
    pred[pred == 6] = 9
    pred[pred == 7] = 9
    pred[pred == 8] = 2
    return mean_IOU_one_sample(pred, gt, 6)


def to_one_hot(target, maxx=50, device_id=0):
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target.astype(np.int64)).cuda(device_id)
    N = target.shape[0]
    target_one_hot = torch.zeros((N, maxx))

    target_one_hot = target_one_hot.cuda(device_id)
    target_t = target.unsqueeze(1)
    target_one_hot = target_one_hot.scatter_(1, target_t.long(), 1)
    return target_one_hot


def matching_iou(matching, predicted_labels, labels):
    """
    Computes the iou score, starting from matching pairs
    coming out from hungarian matching solver. Note that
    it is assumed that iou is only computed over matched pairs.
    That is to say, if any column in the matched pair has zero
    number of points, that pair is not considered.
    """
    batch_size = labels.shape[0]
    IOU = []
    new_pred = []
    for b in range(batch_size):
        iou_b = []
        len_labels = np.unique(predicted_labels[b]).shape[0]
        rows, cols = matching[b]
        count = 0
        for r, c in zip(rows, cols):
            pred_indices = predicted_labels[b] == r
            gt_indices = labels[b] == c

            # if both input and predictions are empty, ignore that.
            if (np.sum(gt_indices) == 0) and (np.sum(pred_indices) == 0):
                continue
            iou = np.sum(np.logical_and(pred_indices, gt_indices)) / (
                        np.sum(np.logical_or(pred_indices, gt_indices)) + 1e-8)
            iou_b.append(iou)

        # find the mean of IOU over this shape
        IOU.append(np.mean(iou_b))
    return np.mean(IOU)


def relaxed_iou(pred, gt, max_clusters=50):
    batch_size, N, K = pred.shape
    normalize = torch.nn.functional.normalize
    one = torch.ones(1).cuda()

    norms_p = torch.sum(pred, 1)
    norms_g = torch.sum(gt, 1)
    cost = []

    for b in range(batch_size):
        p = pred[b]
        g = gt[b]
        c_batch = []
        dots = p.transpose(1, 0) @ g

        for k1 in range(K):
            c = []
            for k2 in range(K):
                r_iou = dots[k1, k2]
                r_iou = r_iou / (norms_p[b, k1] + norms_g[b, k2] - dots[k1, k2] + 1e-7)
                if (r_iou < 0) or (r_iou > 1):
                    import ipdb;
                    ipdb.set_trace()
                c.append(r_iou)
            c_batch.append(c)
        cost.append(c_batch)
    return cost


def relaxed_iou_fast(pred, gt, max_clusters=50):
    batch_size, N, K = pred.shape
    normalize = torch.nn.functional.normalize
    one = torch.ones(1).cuda()

    norms_p = torch.unsqueeze(torch.sum(pred, 1), 2)
    norms_g = torch.unsqueeze(torch.sum(gt, 1), 1)
    cost = []

    for b in range(batch_size):
        p = pred[b]
        g = gt[b]
        c_batch = []
        dots = p.transpose(1, 0) @ g
        r_iou = dots
        r_iou = r_iou / (norms_p[b] + norms_g[b] - dots + 1e-7)
        cost.append(r_iou)
    cost = torch.stack(cost, 0)
    return cost
