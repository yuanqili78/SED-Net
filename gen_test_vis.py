import os
import numpy as np


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from joblib import Parallel, delayed


COLORS_TYPE = np.array([[0, 0, 0],[165, 0, 33],[255, 255, 0],[0, 255, 0], [0, 204, 255],[102, 0, 255],
    [119, 119, 119], [255, 0, 0],[204, 153, 0],[0, 102, 0],[0, 0, 255],[153, 153, 255],
    [221, 221, 221],[255, 153, 153],[255, 204, 102],[0, 204, 153],[153, 204, 255],[153, 0, 255],
    [255, 255, 204],[102, 0, 51],[255, 153, 0],[102, 153, 0],[0, 102, 153],[255, 0, 255],
    [204, 255, 255], [255, 102, 153], [204, 255, 102],[102, 102, 51], [51, 51, 153], [102, 0, 102],
    [204, 204, 0], [255, 204, 255],
[0, 0, 0],[165, 0, 33],[255, 255, 0],[0, 255, 0], [0, 204, 255],[102, 0, 255],
    [119, 119, 119], [255, 0, 0],[204, 153, 0],[0, 102, 0],[0, 0, 255],[153, 153, 255],
    [221, 221, 221],[255, 153, 153],[255, 204, 102],[0, 204, 153],[153, 204, 255],[153, 0, 255],
    [255, 255, 204],[102, 0, 51],[255, 153, 0],[102, 153, 0],[0, 102, 153],[255, 0, 255],
    [204, 255, 255], [255, 102, 153], [204, 255, 102],[102, 102, 51], [51, 51, 153], [102, 0, 102],
    [204, 204, 0], [255, 204, 255]
])


FACTORY_COLORS_TYPE = np.array([[165, 0, 33],[255, 255, 0],[0, 255, 0], [0, 204, 255],[102, 0, 255],
    [119, 119, 119], [255, 0, 0],[204, 153, 0],[0, 102, 0],[0, 0, 255],[153, 153, 255],
    [221, 221, 221],[255, 153, 153],[255, 204, 102],[0, 204, 153],[153, 204, 255],[153, 0, 255],
    [255, 255, 204],[102, 0, 51],[255, 153, 0],[102, 153, 0],[0, 102, 153],[255, 0, 255],
    [204, 255, 255], [255, 102, 153], [204, 255, 102],[102, 102, 51], [51, 51, 153], [102, 0, 102],
    [204, 204, 0], [255, 204, 255], [165, 0, 33],[255, 255, 0],[0, 255, 0], [0, 204, 255],[102, 0, 255],
    [119, 119, 119], [255, 0, 0],[204, 153, 0],[0, 102, 0],[0, 0, 255],[153, 153, 255],
    [221, 221, 221],[255, 153, 153],[255, 204, 102],[0, 204, 153],[153, 204, 255],[153, 0, 255],
    [255, 255, 204],[102, 0, 51],[255, 153, 0],[102, 153, 0],[0, 102, 153],[255, 0, 255],
    [204, 255, 255], [255, 102, 153], [204, 255, 102],[102, 102, 51], [51, 51, 153], [102, 0, 102],
    [204, 204, 0], [255, 204, 255]
])


COLORS_TYPE = COLORS_TYPE.astype(np.float32)


COLORS_TYPE[COLORS_TYPE.shape[0] // 2:, :] = COLORS_TYPE[COLORS_TYPE.shape[0] // 2:, ::-1]  # ============== reverse the left half cmap

COLORS_TYPE_NEW = np.array(
    [[255, 174, 162],[255,126,195],[199,163,242],[53,198,197],[139,211,148],[169,185,241],[174,84,201],[210,204,124],[213,189,179],[116,128,176],[139,232,255],[227,252,186],[141,141,141],[46,117,182],[154,117,0],[136,127,249],[142,186,234],[249,243,155],
[158,179,124],[173,148,146],[204,154,116],[243,132,77],[208,237,83],[187,243,227],[90,156,228],[103,69,211],[146,99,181],[221,83,106],[115,23,38],[235,194,111]]
).astype(np.float32)


def visual_labels(points, label, rand_color_type):

    output = np.zeros((points.shape[0], 6))
    output[:, :3] = points
    for i in range(len(rand_color_type)):
        output[np.where(label == i), 3:] = np.array(rand_color_type[i], dtype=float)
    return output



def gen_vis(src="", id=0):
    types = np.loadtxt(os.path.join(src, f"{id}_type.txt")).astype(np.int32)
    insts = np.loadtxt(os.path.join(src, f"{id}_inst.txt")).astype(np.int32)
    GT_types = np.loadtxt(os.path.join(src, f"{id}_GT_type.txt")).astype(np.int32)
    GT_insts = np.loadtxt(os.path.join(src, f"{id}_GT_inst.txt")).astype(np.int32)
    GT_points = np.loadtxt(os.path.join(src, f"{id}_GT_points.txt"), delimiter=" ")
    
    COLORMAP_VIR_INST = cm.get_cmap('viridis', max(insts.max(), GT_insts.max()))

    return {
        "pred_type": visual_labels(GT_points, types, COLORS_TYPE),
        "pred_inst": visual_labels(GT_points, insts, COLORMAP_VIR_INST(range(insts.max()))[:, :3]),
        "GT_type": visual_labels(GT_points, GT_types, COLORS_TYPE),
        "GT_inst": visual_labels(GT_points, GT_insts, COLORMAP_VIR_INST(range(GT_insts.max()))[:, :3]),
         }


def wrap_fun(src="", id=0, dst=""):
    dict_ret = gen_vis(os.path.join(src, "vis_results"), id)
    for key in dict_ret.keys():
        np.savetxt(os.path.join(dst, f"{id}_{key}.txt"), dict_ret[key], delimiter=";", fmt="%0.4f")


def gen_total_vis(src=""):
    dst = os.path.join(src, "VIS")
    if not os.path.exists(dst):
        os.mkdir(dst)
    
    Parallel(n_jobs=8)(delayed(wrap_fun)(src, test_id, dst) for test_id in range(2699))  # n_jobs=4


if __name__ == "__main__":
    gen_total_vis(src="")