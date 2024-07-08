import math
from math import sqrt

import numpy as np
import torch

from fitting_utils import project_to_plane
from pointnet2.pointnet2_utils import three_nn
from utils import visualize_point_cloud


def bad_points_mask(raw_points, primitives, primitive_ids: np.ndarray, fitter):
    points, insts = torch_type(raw_points), torch_type(primitives)

    print(">>>>> filter bad points")
    bad_mask = torch.zeros((points.shape[0], ), dtype=torch.bool)
    for i in range(primitive_ids.shape[0]):
        # ["plane", axis.reshape((3, 1)), distance]
        # ["cylinder", a, center, radius]
        # ["cone", apex.reshape((1, 3)), axis.reshape((3, 1)), theta]
        # ["sphere", center, radius]
        index = torch.nonzero(insts == torch_type(primitive_ids[i])).squeeze(-1)
        _points = points[index, :]
        print(index.shape)
        if fitter.fitting.parameters[i][0] == "none":
            continue
        elif fitter.fitting.parameters[i][0] == "plane":
            a, d = torch_type(fitter.fitting.parameters[i][1]), torch_type(fitter.fitting.parameters[i][2])
            residual = ((_points @ a).reshape((-1,)) - d).abs()
            bad_idx = residual > 0.05
            bad_idx_origin = index[bad_idx]
            bad_mask[bad_idx_origin] = True

        elif fitter.fitting.parameters[i][0] == "cylinder":
            a, c, r = torch_type(fitter.fitting.parameters[i][1]), torch_type(fitter.fitting.parameters[i][2]), torch_type(fitter.fitting.parameters[i][3])

            residual = (points_to_line_distance(_points, a, c)[0] - r).abs()

            bad_idx = residual > 0.03
            bad_idx_origin = index[bad_idx]
            bad_mask[bad_idx_origin] = True

    return bad_mask.cpu().numpy()

def get_edges_between_insts(points, insts, strict=True):
    points, insts = torch_type(points), torch_type(insts)
    points = points[:, :3]
    three_nn_index = three_nn(torch_type(points).unsqueeze(0), torch_type(points).unsqueeze(0))[1][0]  # N x 3
    one_nn_idx = three_nn_index[:, 1].long() # N
    two_nn_idx = three_nn_index[:, 2].long() # N
    one_nn_inst = torch.gather(insts, index=one_nn_idx, dim=0)
    one_nn_diff_idx = one_nn_inst != insts
    two_nn_inst = torch.gather(insts, index=two_nn_idx, dim=0)
    two_nn_diff_idx = two_nn_inst != insts
    # visualize_point_cloud(points[one_nn_diff_idx & two_nn_diff_idx, :], viz=True)
    if strict:
        return one_nn_diff_idx & two_nn_diff_idx
    else:
        return one_nn_diff_idx


def face_face_inter_map(points, insts, primitive_ids, nn_num_thresh=3):
    """
    primitive_ids：有效instance  num: x
    """
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float().cuda()
    if isinstance(insts, np.ndarray):
        insts = torch.from_numpy(insts).cuda()

    three_nn_index = three_nn(points.unsqueeze(0), points.unsqueeze(0))[1][0]  # N x 3
    one_nn_idx = three_nn_index[:, 1].long() # N
    two_nn_idx = three_nn_index[:, 2].long() # N

    # visualize_point_cloud(points[one_nn_idx, :], viz=True)

    mat = torch.zeros((30, 30), dtype=torch.bool)  # 30 X 30
    for i, _id in enumerate(primitive_ids):
        inst_idx = insts == _id  # N

        nn_idx = one_nn_idx[inst_idx]
        nn_inst = insts[nn_idx]
        diff_insts = nn_inst[nn_inst != _id]  # [2,3,3,4,7,...]

        nn_idx = two_nn_idx[inst_idx]
        nn_inst = insts[nn_idx]
        diff_insts = torch.cat((diff_insts, nn_inst[nn_inst != _id]), dim=0)

        diff_insts_set, counts = torch.unique(diff_insts, return_counts=True)
        diff_insts_set = diff_insts_set.long()
        _id = _id.long()
        for i in range(diff_insts_set.shape[0]):
            if counts[i] >= nn_num_thresh:   # ====================================================================
                mat[_id, diff_insts_set[i]] = True

    for i in range(mat.shape[0]):
        if mat[i, :].max() == False and i in primitive_ids:
            """
            特殊情况
            一个面无相交, inst编号为i
            """
            inst_idx = insts == i
            sample_points = points[inst_idx, :]
            dis_map = (points - sample_points[0, :].unsqueeze(0)).pow(2).sum(-1)  # N
            dis_sort = torch.argsort(dis_map)  # N

            inst_sort = insts[dis_sort]
            first_nn_inst_id = inst_sort[inst_sort != i][0]
            print("lonely inst {} neib is {}".format(i, int(first_nn_inst_id)))
            mat[i, first_nn_inst_id] = mat[i, first_nn_inst_id] = True
    return mat

def np_process(*a):
    a = list(a)
    for i in range(len(a)):
        if isinstance(a[i], torch.Tensor):
            a[i] = a[i].detach().cpu().numpy()

        if isinstance(a[i], np.ndarray) and a[i].shape in [(1, 3), (3, 1)]:
            a[i] = a[i].reshape((3,))

        if isinstance(a[i], np.ndarray) and a[i].shape == ():
            a[i] = a[i].item()

    return a


def vector_product(a1, a2):
    """
    叉积运算
    """
    a1, a2 = np_process(a1, a2)
    a = np.cross(a1, a2)
    return a / np.linalg.norm(a)

def vector_cos(a1, a2):
    a1, a2 = np_process(a1, a2)
    up = a1.dot(a2)
    down = np.linalg.norm(a1) * np.linalg.norm(a2) + 1e-8
    return up / down

def plane_plane_inter_line(face1, face2):
    # ["plane", axis.reshape((3, 1)), distance]
    # ["cylinder", a, center, radius]
    # ["cone", apex.reshape((1, 3)), axis.reshape((3, 1)), theta]
    # ["sphere", center, radius]
    a1 = face1[1].reshape((3, )).cpu().numpy()
    a2 = face2[1].reshape((3, )).cpu().numpy()

    if vector_cos(a1, a2) >= 0.98:
        print("plane parallel!!")
        return "None", None, None

    d1 = face1[2].cpu().numpy()
    d2 = face2[2].cpu().numpy()
    direction = vector_product(a1, a2)

    base = []

    # 假定 z = 0
    A = np.array([[a1[0], a1[1]], [a2[0], a2[1]]])
    if np.linalg.matrix_rank(A) == A.shape[0]:
        xy = np.linalg.inv(A) @ np.array([d1, d2]).reshape((2, 1))
        base.append(np.array([xy[0, 0], xy[1, 0], 0.0]))
    # 假定 x = 0
    A = np.array([[a1[1], a1[2]], [a2[1], a2[2]]])
    if np.linalg.matrix_rank(A) == A.shape[0]:
        yz = np.linalg.inv(A) @ np.array([d1, d2]).reshape((2, 1))
        base.append(np.array([0.0, yz[0, 0], yz[1, 0]]))
    # 假定 y = 0
    A = np.array([[a1[0], a1[2]], [a2[0], a2[2]]])
    if np.linalg.matrix_rank(A) == A.shape[0]:
        xz = np.linalg.inv(A) @ np.array([d1, d2]).reshape((2, 1))
        base.append(np.array([xz[0, 0], 0.0, xz[1, 0]]))
    return "line", direction, base


def cylinder_cylinder_inter_line(face1, face2):
    a1, center1 , radius1, a2, center2, radius2 = np_process(face1[1], face1[2], face1[3], face2[1], face2[2], face2[3])

    center2_proj_axis1 = points_project_to_line(center2, a1, center1)[0].cpu().numpy().reshape((3,))
    dis = np.linalg.norm(center2_proj_axis1-center2)
    if dis > radius2 + radius1 or \
            np.abs(radius2 - radius1) > dis :
        return "None", None

    elif np.abs(vector_cos(a1, a2)) > 1 - 1e-2 and np.abs(radius2 - radius1) < 3e-2:
        return "None", None
    else:
        # TODO
        # 相贯线
        # =========================================================================================================================
        return "None", None



def cylinder_cylinder_inter_line(face1, face2):
    """
    face1: plane: axis.reshape((3, 1)), distance
    face2: cylinder: a, center, radius
    """
    a1 = face1[1].reshape((3, )).cpu().numpy()
    d1 = face1[2].cpu().numpy()
    a2 = face2[1].reshape((3, )).cpu().numpy()
    center = face2[2].reshape((3, )).cpu().numpy()
    radius = face2[3].cpu().numpy() if isinstance(face2[3], torch.Tensor) else face2[3]
    cos = vector_cos(a1, a2)
    # 圆柱axis和面法向量垂直
    if abs(cos) <= 1.5e-2:
        print("plane 和 cylinder 交线为 直线")
        direction = a2

        proj_center = project_to_plane(face2[2].reshape((1, 3)), face1[1], face1[2]).cpu().numpy()
        # print(proj_center)

        tmp = np.power(radius, 2) - np.power(proj_center - center, 2).sum()
        if tmp < -1e-3:
            print("palne and cylinder not intersect!")
            return "None", None

        if np.abs(tmp) <= 1e-3:
            print("面与圆相切!")
            proj_dir = vector_product(a1, a2)
            return "line", direction, proj_center

        pro_d = np.sqrt(np.power(radius, 2) - np.power(proj_center - center, 2).sum())
        # center的圆截面与face1平面的交线
        proj_dir = vector_product(a1, a2)
        print(pro_d)
        base_1 = proj_center + pro_d * proj_dir.reshape((1, -1))
        base_2 = proj_center - pro_d * proj_dir.reshape((1, -1))

        return "two line", direction, base_1, base_2

    elif 1 - abs(cos) <= 1e-2:
        print("plane inter cylinder get circle")
        axis = a2
        proj_center = project_to_plane(face2[2].reshape((1, 3)), face1[1], face1[2]).cpu().numpy()

        # 获取参数方程的x_axis, y_axis
        my_axis = get_circle_x_y_axis(axis)

        return "circle", proj_center, my_axis, radius

    else:
        print("plane inter cylinder 椭圆")
        proj_center = project_to_plane(face2[2].reshape((1, 3)), face1[1], face1[2]).cpu().numpy()

        # 求解椭圆中心
        A = np.zeros((4, 4), dtype=np.float64)
        A[1:, :3] = np.diag([1.0, 1.0, 1.0])
        A[3, 1:] = -a2
        A[0, :3] = a1
        Y = np.array([d1, center[0], center[1], center[2]]).reshape((4, 1))
        X = np.linalg.inv(A) @ Y
        ellipse_center = X[:3, 0].reshape((3, ))
        x_axis = ellipse_center - proj_center
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = vector_product(x_axis, center - ellipse_center)
        x_radius = radius / (abs(cos) + 1e-8)
        y_radius = radius

        return 'ellipse', ellipse_center, x_axis, y_axis, x_radius, y_radius


def plane_cone_inter_line(face1, face2):
    # ["plane", axis.reshape((3, 1)), distance]
    # ["cone", apex.reshape((1, 3)), axis.reshape((3, 1)), theta]
    a1, a2, apex = np_process(face1[1], face2[2], face2[1])  # 3 ,3 ,3
    theta = face2[-1].cpu().numpy()

    cos = vector_cos(a1, a2)

    if np.abs(cos) >= 0.98:
        print("line cone 正交")
        proj_center = project_to_plane(face2[1].reshape((1, 3)), face1[1], face1[2]).cpu().numpy()
        print(proj_center)
        apex_projC_dis = np.linalg.norm(proj_center - apex)
        radius = apex_projC_dis * np.tan(theta)
        xy_axis = get_circle_x_y_axis(a2)

        return "circle", proj_center, xy_axis, radius

    else:
        return "", None


def cylinder_cone_inter_line(face1, face2):
    # ["cylinder", a, center, radius]
    # ["cone", apex.reshape((1, 3)), axis.reshape((3, 1)), theta]
    a1, center1 , radius1, a2, apex, theta = np_process(face1[1], face1[2], face1[3], face2[2], face2[1], face2[-1])

    cos = vector_cos(a1, a2)

    if np.abs(cos) >= 0.98:
        print("cylinder cone 正交")
        apex_projC_dis = radius1 / np.tan(theta)
        proj_center = apex + a2 * apex_projC_dis
        if vector_cos(apex - proj_center, a2) < 0:
            proj_center = apex - a2 * apex_projC_dis
        xy_axis = get_circle_x_y_axis(a1)

        return "circle", proj_center, xy_axis, radius1

    else:
        return "", None


def plane_sphere_inter_line(face1, face2):
    """
    center [1, 3] radius
    """
    a, d, center, radius = np_process(face1[1], face1[2], face2[1], face2[2])
    center_proj = project_to_plane(face2[1].reshape((1, 3)), face1[1], face1[2]).cpu().numpy()
    center_plane_dis = np.linalg.norm(center_proj - center)
    if center_plane_dis >= radius:
        return "", None
    else:
        radius1 = sqrt(radius ** 2 - center_plane_dis.item() ** 2)
        xy_axis = get_circle_x_y_axis(a)
        return "circle", center_proj, xy_axis, radius1





def cylinder_sphere_inter_line(face1, face2):

    a1, center1, radius1, center, radius = np_process(face1[1], face1[2], face1[3], face2[1], face2[2])
    center_proj = center
    xy_axis = get_circle_x_y_axis(a1)
    return "circle", center_proj, xy_axis, radius1


def get_circle_x_y_axis(axis):
    A = np.array([[axis[0],0,0], [0,axis[1],0], [0,0, axis[2]]], dtype=np.float64)
    u, s, v = np.linalg.svd(A)
    x_axis = v[-1, :].reshape((3,))
    y_axis = vector_product(x_axis, axis)
    return [x_axis / np.linalg.norm(x_axis), y_axis / np.linalg.norm(y_axis)]


def torch_type(d):
    if isinstance(d, (np.ndarray, )):
        d = torch.from_numpy(d)
    if isinstance(d, (np.float64, np.int32)):
        d = torch.Tensor([d,])[0]

    return d.float().contiguous().cuda()


def points_project_to_line(points, k, d):
    points, k, d = torch_type(points), torch_type(k), torch_type(d)
    p2l_dis, proj_dis = points_to_line_distance(points, k, d)  # N, N
    proj_points_1 = d.reshape((1, 3)) + proj_dis @ k.reshape((1, -1))
    proj_points_2 = d.reshape((1, 3)) - proj_dis @ k.reshape((1, -1))
    dis_1 = torch.norm(points - proj_points_1, 2)
    dis_2 = torch.norm(points - proj_points_2, 2)
    final_proj_points = torch.zeros_like(proj_points_1)
    final_proj_points[dis_1 <= dis_2, :] = proj_points_1
    final_proj_points[dis_1 > dis_2, :] = proj_points_2
    return final_proj_points, p2l_dis.squeeze(-1)


def points_to_line_distance(points, k, d):
    points, k, d = torch_type(points), torch_type(k), torch_type(d)
    tmp = points - d.reshape((1, 3))  # N x 3
    dis = tmp @ k.reshape((3, 1)) / torch.norm(k, 2) # N x 1  投影长度

    sqrt_tmp = tmp.pow(2).sum(-1) - (dis.squeeze(-1)).pow(2)
    sqrt_tmp[(sqrt_tmp <= 0) & (sqrt_tmp >= -1e-3)] = 0

    return sqrt_tmp.sqrt(), dis


def line_line_inter(k1, d1, k2, d2):
    """
    k1 @ n + d1 = xyz
    k2 @ n + d2 = xyz
    """
    A = np.zeros((6, 5), dtype=np.float64)
    A[:3, 0] = k1.T
    A[3:, 1] = k2.T
    A[:3, 2:] = np.diag([-1.0, -1.0, -1.0])
    A[3:, 2:] = np.diag([-1.0, -1.0, -1.0])
    Y = - np.concatenate((d1, d2), axis=0).reshape((6, 1))

    Y, A = torch_type(Y), torch_type(A)
    # print(A.shape, Y.shape)
    X, residual = torch.lstsq(Y, A=A)
    # print(torch.dist(X, torch.linalg.pinv(A) @ Y))

    if X.reshape((6, ))[2:5].abs().max() <= 1.1:
        return X.reshape((6, ))[2:5].cpu().numpy()
    else:
        print("line line don't intersect")
        return None


def line_circle_inter(para1, para2):
    k, d = para1
    center, x_axis, y_axis, radius = para2

    N_circel = vector_product(x_axis, y_axis)
    cos = vector_cos(k, N_circel)
    # 求解圆所在平面
    k_CP = N_circel
    d_CP = (N_circel * center).sum()

    my_A = np.zeros((3, 3))
    my_A[:, 0] = k
    my_A[:, 1] = x_axis
    my_A[:, 2] = y_axis
    if np.linalg.matrix_rank(my_A) == 2:
        print("直线和圆在一个平面")
        c_proj, proj_dis = points_project_to_line(center.reshape((1, 3)), k, d)
        c_proj, proj_dis = np_process(c_proj, proj_dis)
        if proj_dis > radius + 5e-3:
            return None
        elif np.abs(radius - proj_dis) <= 5e-3:
            return np_process(c_proj,)

        else:
            my_d = np.sqrt(radius ** 2 - proj_dis ** 2)
            return np_process(c_proj + my_d * k.reshape((3, )), c_proj - my_d * k.reshape((3, )))

    # 直线上一点在圆平面投影点
    d_proj_circle_plane = project_to_plane(d.reshape((1, 3)), k_CP, d_CP).cpu().numpy().reshape((3,))

    if np.linalg.norm(d_proj_circle_plane - d) <= 1e-2:
        d_proj_circle_plane = project_to_plane((k * 0.1 + d).reshape((1, 3)), k_CP, d_CP).cpu().numpy().reshape((3,))

    if np.linalg.norm(d_proj_circle_plane - center) <= 1e-2:
        d_proj_circle_plane = project_to_plane((k * 0.2 + d).reshape((1, 3)), k_CP, d_CP).cpu().numpy().reshape((3,))

    # 求经过直线且垂直圆的平面
    k_ver = vector_product(d_proj_circle_plane - d, k) if np.abs(cos) > 1e-2 else vector_product(N_circel, k)
    d_ver = (k_ver * d).sum()
    # 圆心在该平面的投影点
    center_proj_ver = project_to_plane(center.reshape((1, 3)), k_ver, d_ver).cpu().numpy().reshape((3,))

    tmp = np.power(radius, 2) - np.power(center_proj_ver - center, 2).sum()
    if tmp < -1e-3:
        return None
    plane_plane_inter_d = np.sqrt(tmp if tmp >= 0 else 0)

    proj_dir = vector_product(center_proj_ver - center, d_proj_circle_plane - d)

    base_1 = center_proj_ver + plane_plane_inter_d * proj_dir.reshape((1, -1))
    base_2 = center_proj_ver - plane_plane_inter_d * proj_dir.reshape((1, -1))

    '''
    res = []
    if filter(base_1, insts_points):
        res.append(base_1)
    if filter(base_2, insts_points):
        res.append(base_2)

    return tuple(res)
    '''

    if np.abs(cos) < 3e-2:
        # mytmp = np.abs(vector_cos(base_1-center, k))
        # if np.abs(vector_cos(base_1-center, k)) < 3e-2:  # ===========================================================
        print("one inter")
        if np.abs(vector_cos(base_1-d, k)) > 0.995:
            return (base_1,)
        elif np.abs(vector_cos(base_2-d, k)) > 0.995:
            return (base_2,)
        else:
            return None
    else:
        print("two inter")
        if np.abs(vector_cos(base_1-d, k)) > 0.995:
            return (base_1, base_2)
        else:
            return None


def fitter_point(point, insts:tuple, thresh=0.01):
    point= torch_type(point)
    for inst in insts:
        inst = torch_type(inst)
        min_dis = (point.reshape((1, 3)) - inst).pow(2).sum(-1).min()  # .sqrt()
        if min_dis > thresh:   # 0.03  ===============================================================================================================
            return False

    return True


def get_line_point_d(k, d, point):
    k, d, point = k.reshape((3,)), d.reshape((3,)), point.reshape((3,))
    for i in range(3):
        if np.abs(k[i]) > 1e-2:
            return (point[i] - d[i]) / k[i]
    return 1


def get_circle_two_point_theta(point1, point2, center, x_axis, y_axis, radius, insts_points:tuple):
    point1, point2, center, x_axis, y_axis, radius = torch_type(point1), torch_type(point2), \
                                                     torch_type(center), torch_type(x_axis), \
                                                     torch_type(y_axis), torch_type(radius)
    angle1 = get_circle_point_theta(point1, center, x_axis, y_axis, radius)
    angle2 = get_circle_point_theta(point2, center, x_axis, y_axis, radius)

    angle_1, angle_2 = min(angle1, angle2), max(angle1, angle2)
    mid = (angle_1 + angle_2) / 2
    sample_mid = radius * (math.cos(mid) * x_axis + math.sin(mid) * y_axis) + center
    if fitter_point(sample_mid, insts_points):
        return np_process(angle_1, angle_2)

    else:
        pi = torch.acos(torch.zeros(1)).item() * 2
        return np_process(angle_2, angle_1 + 2 * pi)


def get_circle_point_theta(point, center, x_axis, y_axis, radius):
    # radius * (np.cos(theta_1) * x_axis + np.sin(theta_1) * y_axis) + center
    point, center, x_axis, y_axis, radius = torch_type(point), torch_type(center), torch_type(x_axis), torch_type(y_axis), torch_type(radius)

    A = torch.cat((x_axis.view(3, 1), y_axis.view(3, 1)), dim=1)
    Y = ((point - center) / radius).view(3, 1)
    X, _ = torch.lstsq(Y, A=A)
    _cos, _sin = X[0, 0], X[1, 0]
    print("cos", _cos, "sin", _sin)
    _cos = torch.clamp(_cos, -1, 1)
    _sin = torch.clamp(_sin, -1, 1)

    theta_cos=torch.acos(_cos)
    theta_sin=torch.asin(_sin)
    pi = torch.acos(torch.zeros(1)).item() * 2
    if abs(pi - theta_sin - theta_cos) < 3e-2 or \
            abs(theta_sin - theta_cos) < 3e-2 or \
            abs(theta_sin - theta_cos + 2 * pi) < 3e-2 or \
            abs(pi - theta_sin - theta_cos + pi * 2) < 3e-2:
        return theta_cos
    theta_cos_2 = - theta_cos + 2 * pi
    if abs(pi - theta_sin - theta_cos_2) < 2e-2 or abs(theta_sin - theta_cos_2) < 2e-2 or \
            abs(pi - theta_sin - theta_cos_2 + pi * 2) < 2e-2:
        return theta_cos_2

    theta_cos_3 = - theta_cos
    if abs(pi - theta_sin - theta_cos_3) < 3e-2 or \
            abs(theta_sin - theta_cos_3) < 3e-2 or \
            abs(theta_sin - theta_cos_3 + 2 * pi) < 3e-2 or \
            abs(pi - theta_sin - theta_cos_3 + pi * 2) < 3e-2:
        return theta_cos_3
    '''
    theta_tan = 0
    if _cos.abs() > 2e-3:
        _tan = _sin / _cos
        theta_tan = torch.atan(_tan)

    else:
        _tanh = _cos / _sin
        theta_tan = torch.atan(_tanh)
    # a, b = torch.asin(_sin), torch.acos(_cos)
    # print(a, b)
    theta_cos = torch.acos(_cos)
    print(theta_tan, theta_cos)
    pi = torch.acos(torch.zeros(1)).item() * 2
    if abs(theta_tan + pi - theta_cos) < 1e-2 or abs(theta_tan - theta_cos) < 1e-2:
        return theta_cos

    theta_cos_2 = - theta_cos + 2 * pi
    if abs(theta_tan + pi - theta_cos_2) < 1e-2 or abs(theta_tan - theta_cos_2) < 1e-2:
        return theta_cos_2
    '''
    return 0

def plane_cylinder_inter_line(face1, face2, plane_sample=None):
    """
    face1: plane: axis.reshape((3, 1)), distance
    face2: cylinder: a, center, radius
    """
    a1 = face1[1].reshape((3, )).cpu().numpy()
    d1 = face1[2].cpu().numpy()
    a2 = face2[1].reshape((3, )).cpu().numpy()
    center = face2[2].reshape((3, )).cpu().numpy()
    radius = face2[3].cpu().numpy() if isinstance(face2[3], torch.Tensor) else face2[3]
    cos = vector_cos(a1, a2)
    # 圆柱axis和面法向量垂直
    if abs(cos) <= 1.5e-2:
        print("plane 和 cylinder 交线为 直线")
        direction = a2

        proj_center = project_to_plane(face2[2].reshape((1, 3)), face1[1], face1[2]).cpu().numpy()
        # print(proj_center)

        tmp = np.power(radius, 2) - np.power(proj_center - center, 2).sum()
        if tmp < -1e-3:
            print("palne and cylinder not intersect!")
            return "None", None

        if np.abs(tmp) <= 1e-3:
            print("面与圆相切!")
            proj_dir = vector_product(a1, a2)
            return "line", direction, proj_center

        pro_d = np.sqrt(np.power(radius, 2) - np.power(proj_center - center, 2).sum())
        # center的圆截面与face1平面的交线
        proj_dir = vector_product(a1, a2)
        print(pro_d)
        base_1 = proj_center + pro_d * proj_dir.reshape((1, -1))
        base_2 = proj_center - pro_d * proj_dir.reshape((1, -1))

        return "two line", direction, base_1, base_2

    elif 1 - abs(cos) <= 1e-2:
        print("plane inter cylinder get circle")
        axis = a2
        proj_center = project_to_plane(face2[2].reshape((1, 3)), face1[1], face1[2]).cpu().numpy()

        # 获取参数方程的x_axis, y_axis，SVD可能出现数值计算误差
        if plane_sample is not None:
            my_axis = [plane_sample - proj_center.reshape(3,), ]
            my_axis[0] = my_axis[0] / np.linalg.norm(my_axis[0])
            my_axis.append(vector_product(my_axis[0], axis))
        else:
            my_axis = get_circle_x_y_axis(axis)


        return "circle", proj_center, my_axis, radius

    else:
        print("plane inter cylinder 椭圆")
        proj_center = project_to_plane(face2[2].reshape((1, 3)), face1[1], face1[2]).cpu().numpy()    # 正确的

        # 求解椭圆中心
        '''
        A = np.zeros((4, 4), dtype=np.float64)
        A[1:, :3] = np.diag([1.0, 1.0, 1.0])
        A[3, 1:] = -a2
        A[0, :3] = a1
        Y = np.array([d1, center[0], center[1], center[2]]).reshape((4, 1))
        X = np.linalg.inv(A) @ Y
        ellipse_center = X[:3, 0].reshape((3, ))'''
        t = -(np.dot(a1, center) - d1) / (np.dot(a1, a2)+1e-10)
        ellipse_center = center + t * a2

        x_axis = ellipse_center - proj_center

        # debug
        save = np.arange(0, 1, step=0.005).reshape(-1, 1)
        save = save * ellipse_center.reshape(1, 3) + (1 - save) * proj_center.reshape(1, 3)
        np.savetxt("tmp.txt", save, delimiter=";", fmt="%0.3f")

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = vector_product(x_axis, center - ellipse_center)
        x_radius = radius / (abs(cos) + 1e-8)
        y_radius = radius

        return 'ellipse', ellipse_center, x_axis, y_axis, x_radius, y_radius

if __name__ == "__main__":
    face1 = ['', torch.rand((3, 1)), torch.zeros((1, ))[0]]
    face2 = ['', torch.rand((3, 1)), torch.zeros((1,))[0]]
    print(plane_plane_inter_line(face1, face2))

