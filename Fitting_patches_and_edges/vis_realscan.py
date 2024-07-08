import os
import numpy as np

id = "1347_MechanicalPin_clean"

_dir = "new_datas/real_scan"
save_dir = "aaa"


vis_I = np.loadtxt(_dir + "/" + id + "_Vis_I.txt", delimiter=";")

primitives = np.load("./{}/{}_inst.npy".format(_dir, id)).astype(np.int32)


color = vis_I[primitives == 4, 3:][0]
vis_I[primitives == 8, 3:] = color
# vis_I[primitives == 0, 3:] = color
np.savetxt(save_dir + "/" + id + "_Vis_I.txt", vis_I, delimiter=";", fmt="%0.4f")