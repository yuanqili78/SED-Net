import os
import numpy as np
from pykdtree.kdtree import KDTree
import torch

def npy(var):
    return var.data.cpu().numpy()

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_idx(x, k):
    """
    x: [B, N, 3]
    """
    dis_maps = square_distance(x, x)  # [B, N, N]

    idx = dis_maps.topk(k=k, dim=-1)[1]

    return idx


def construction_affinity_matrix_normal(inputs_xyz, N_gt, sigma=0.1, knn=50):
    '''
    inputs_xyz: (B, N, 3)
    N_gt: (B, N, 3)

    check distance function

    '''
    #TODO: add embedding weight, might be useful
    
    B, N, _ = N_gt.shape
    normal = N_gt.transpose(1, 2).contiguous()
    affinity_matrix = torch.zeros(B, N, N).float().to(N_gt.device)

    '''
    nnid = []
    for b in range(N_gt.shape[0]):

        pc = npy(inputs_xyz[b]).T
        tree = KDTree(pc)
        nndst_, nnid_ = tree.query(pc, k=knn)
        nnid.append(torch.from_numpy(nnid_.astype('float')).long().to(N_gt.device))
    nnid = torch.stack(nnid)
    k = nnid.shape[-1]'''

    nnid = knn_idx(inputs_xyz, knn)
    k = nnid.shape[-1]
    
    # xyz_sub = torch.gather(inputs_xyz.view(B, 3, -1), -1, nnid.view(B, 1, -1).repeat(1, 3, 1)).view(B, 3, -1, k)  # [b, 3, N, k]
 
    n_sub = torch.gather(normal, -1, nnid.view(B, 1, -1).repeat(1, 3, 1)).view(B, 3, -1, k)
    
    dst = torch.acos((normal.unsqueeze(-1) * n_sub).sum(1).clamp(-0.99, 0.99))
    # index = nnid
    dst = torch.exp(-dst**2 / (2 * sigma * sigma))
    
    affinity_matrix = affinity_matrix.scatter_add(-1, nnid, dst)
    background_mask = affinity_matrix == 0
    affinity_matrix = affinity_matrix + 0

    affinity_matrix[background_mask] = 1e-12
   
    D = affinity_matrix.sum(-1)
    D = torch.diag_embed(1.0 / D.sqrt())
    affinity_matrix = torch.matmul(torch.matmul(D, affinity_matrix), D)
    
    mask = (affinity_matrix > 0).float()
    affinity_matrix = (affinity_matrix + affinity_matrix.permute(
        0, 2, 1)) / (mask + mask.permute(0, 2, 1)).clamp(1, 2)
 
    return affinity_matrix


def compute_entropy(features, CHUNK=2000):
    '''
    features: (1, N, K)   K = dim of feature
    '''
    
    eps = 1e-7
    assert(features.shape[0] == 1)

    # ==================

    ITER = 5
    # ==================
    
    feat = features[0]

    N, K = feat.shape
    
    average_dst = 0
    
    # calculate interval
    max_list = []
    min_list = []
    
    # save gpu memory
    for i in range(ITER):
        for j in range(ITER):
            max_ = torch.max((feat[i*CHUNK:(i+1)*CHUNK, None, :] - feat[None, j*CHUNK:(j+1)*CHUNK, :]).view(-1,K), dim=0)[0][None, :]
            min_ = torch.min((feat[i*CHUNK:(i+1)*CHUNK, None, :] - feat[None, j*CHUNK:(j+1)*CHUNK, :]).view(-1,K), dim=0)[0][None, :]
            max_list.append(max_)
            min_list.append(min_)
    
    max_all = torch.max(torch.cat(max_list, dim=0), dim=0)[0]
    min_all = torch.min(torch.cat(min_list, dim=0), dim=0)[0]
    interval = max_all - min_all

    # calculate average_dst
    for i in range(ITER):
        for j in range(ITER):
            dst = torch.norm((feat[i*CHUNK:(i+1)*CHUNK, None, :] - feat[None, j*CHUNK:(j+1)*CHUNK, :]) / interval, dim=2)

            average_dst += torch.sum(dst)
    
    average_dst /= (N*N)
    
    alpha = -np.log(0.5) / average_dst
    
    E = 0

    for i in range(ITER):
        for j in range(ITER):
            dst = torch.norm((feat[i*CHUNK:(i+1)*CHUNK, None, :] - feat[None, j*CHUNK:(j+1)*CHUNK, :]) / interval, dim=2)
            s = torch.exp(-alpha * dst)

            entropy = - s * torch.log(s + eps) - (1 - s) * torch.log(1 - s + eps)

            E += torch.sum(entropy)

    E /= (N*N)

    return E


def hpnet_process(affinity_feat, inputs_xyz, normals, id=None, types=None, edges=None, normal_smooth_w=0.5, CHUNK=2000, gpu='cuda:0', drop_rest_idx=None):
    """
    affinity_feat: (B, N, K)  without L2
    inputs_xyz: (B, N, 3)
    normals: (B, N, 3)
    types: (B, N, 6) or None
    """
    spec_embedding_list = []
    weight_ent = []

    # ===========================
    # instance embedding 
    feat_ent_weight = 1.7
    
    '''
    feat_ent = 0
    for i in range(affinity_feat.shape[0]):
        feat_ent += feat_ent_weight - float(compute_entropy(affinity_feat[0].unsqueeze(0)))
    feat_ent /= affinity_feat.shape[0]'''
            
    feat_ent = feat_ent_weight - float(compute_entropy(affinity_feat, CHUNK=CHUNK))
    weight_ent.append(feat_ent)
    spec_embedding_list.append(affinity_feat)

    # ===========================
    # normal smooth embedding
    
    edge_topk = 12
    normal_sigma = 0.1
    edge_knn = 50
    edge_ent_weight = normal_smooth_w 

    fn = "src/normal_smooth_cache/Us_{}_{}_{}.pt".format(id, normal_sigma, edge_knn)
    fn_ent = "src/normal_smooth_cache/WUs_{}_{}_{}.pt".format(id, normal_sigma, edge_knn)
    if id is not None and os.path.exists(fn) and os.path.exists(fn_ent):
        print("{} Us find cache".format(id))
        v = torch.load(fn).cuda(gpu)  # [1, 10000, 12]
        ent = torch.load(fn_ent)
        # print(v.shape, ent.shape)
    else:
        affinity_matrix_normal = construction_affinity_matrix_normal(inputs_xyz, normals, sigma=normal_sigma, knn=edge_knn) 
        v = torch.lobpcg(affinity_matrix_normal, k=edge_topk, niter=10)[1]
        v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-16)
        torch.save(v, fn)
        ent = compute_entropy(v, CHUNK=CHUNK)
        torch.save(ent, fn_ent)


    if drop_rest_idx is not None:
        v = v[:, drop_rest_idx, :]

    edge_ent = edge_ent_weight - float(ent)
           
    weight_ent.append(edge_ent)
    spec_embedding_list.append(v)

    # ====== concat type 6c cls
    if types is not None:
        types = torch.exp(types)  # [0 ~ 1]
        if edges is not None:
            types = torch.cat((types, torch.softmax(edges, dim=-1)), dim=-1)
        # print(types.shape, types.max())
        types_ent_weight = 0.25   # ==============================================
        type_ent = types_ent_weight - float(compute_entropy(types, CHUNK=CHUNK))
        weight_ent.append(type_ent)
        spec_embedding_list.append(types)        

    # ===========================
    # combine all
    weighted_list = []
    # norm_weight_ent = weight_ent / np.linalg.norm(weight_ent)
    for i in range(len(spec_embedding_list)):
        weighted_list.append(spec_embedding_list[i] * weight_ent[i])

    spectral_embedding = torch.cat(weighted_list, dim=-1)

    return spectral_embedding