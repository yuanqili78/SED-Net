from mean_shift import MeanShift
import torch
import numpy as np
import torch.nn.functional as F

from proj_2_edge_utils import get_edges_between_insts, torch_type, np_process

mean_shift = MeanShift()

color_dict_30 = np.array([[0, 0, 0],[165, 0, 33],[255, 255, 0],[0, 255, 0], [0, 204, 255],[102, 0, 255],
    [119, 119, 119], [255, 0, 0],[204, 153, 0],[0, 102, 0],[0, 0, 255],[153, 153, 255],
    [221, 221, 221],[255, 153, 153],[255, 204, 102],[0, 204, 153],[153, 204, 255],[153, 0, 255],
    [255, 255, 204],[102, 0, 51],[255, 153, 0],[102, 153, 0],[0, 102, 153],[255, 0, 255],
    [204, 255, 255], [255, 102, 153], [204, 255, 102],[102, 102, 51], [51, 51, 153], [102, 0, 102],
    [204, 204, 0], [255, 204, 255]])


def visual_labels(points, label, rand_color_type):
    points, label = np_process(points, label)
    output = np.zeros((points.shape[0], 6))
    output[:, :3] = points
    for i in range(len(rand_color_type)):
        output[np.where(label == i), 3:] = np.array(rand_color_type[i], dtype=float) / 255
    return output


def insts_cluster(src, dst, _id, cluster, ratio_thresh=0.15):
    primitives = np.loadtxt("./{}/None_{}_inst.txt".format(src, _id)).astype(np.int32)
    types = np.loadtxt("./{}/None_{}_type.txt".format(src, _id)).astype(np.int32)
    raw_points = np.loadtxt("./{}/{}_gt_points.txt".format(src, _id), delimiter=";")
    edge_confi = np.loadtxt("./{}/None_{}_edge_confi.txt".format(src, _id), delimiter=";")
    is_edge = edge_confi.argmax(-1) == 1  # N x 1

    N_num = raw_points.shape[0]

    inst_data = []
    primitive_ids = np.unique(primitives)
    print(primitive_ids)

    strict_edge_idx = get_edges_between_insts(raw_points, primitives).cpu().numpy()  # 得到inst之间的边界点索引

    for inst in primitive_ids:
        index = primitives == inst  # N,

        count = np.bincount(types[index])
        labels = np.argmax(count)

        if labels == 2: # cylinder
            # index = index & ~is_edge  # 除去边界，使得拟合更稳定
            index = index & ~strict_edge_idx

        if labels == 3:  # Cone
            index = index & ~strict_edge_idx

        points = raw_points[index, :3]
        normals = raw_points[index, 3:]

        inst_data.append((torch.from_numpy(points).cuda(),
                     torch.from_numpy(normals).cuda(),
                     labels, index))

    _mask = np.ones((30, ), dtype=bool)
    _mask[primitive_ids] = False
    potential_prim_ids = np.arange(30)[_mask]
    next_new_id = 0

    print(potential_prim_ids)
    new_primitive_ids = []
    new_inst_data = []
    # new_primitives = primitives

    for i in range(len(inst_data)):
        if inst_data[i][0].shape[0] < N_num * ratio_thresh:
            new_primitive_ids.append(primitive_ids[i])
            new_inst_data.append(inst_data[i])

        else:
            if next_new_id >= potential_prim_ids.shape[0]:
                new_primitive_ids.append(primitive_ids[i])
                new_inst_data.append(inst_data[i])
                continue

            labels = torch_type(F.one_hot(torch_type(types)[inst_data[i][-1]].long(), 5))
            points = inst_data[i][0]
            normals = inst_data[i][1]
            cur_inst_id = torch_type(primitive_ids[i])

            features = torch.cat((normals, points, labels), 1)
            features = F.normalize(features)

            print(features.shape)

            wb = 0.5
            iterations=25

            new_X, center, bw, new_labels = mean_shift.mean_shift(features, features.shape[0] // 4,
                                             wb, iterations=iterations,
                                             nms=True)
            print(new_X.shape)
            # print(np_process(features[:10,:]))
            # print(np_process(new_X[:10,:]))
            np.savetxt("{}/{}_debug_total_{}_{}.txt".format(dst, _id, wb, iterations),
                       visual_labels(points, new_labels, color_dict_30),
                       fmt="%0.4f", delimiter=';')


if __name__=="__main__":
    _dir = "cluster_src"
    save_dir = "cluster_dst"
    id = 26
    insts_cluster(_dir, save_dir, id, mean_shift, ratio_thresh=0.15)
