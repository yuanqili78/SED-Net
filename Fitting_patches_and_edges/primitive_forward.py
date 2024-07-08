import traceback

from open3d import *
import numpy as np
import torch
from geomdl import fitting as geomdl_fitting
from lapsolver import solve_dense
from open3d import *
from open3d import *

from VisUtils import tessalate_points
from approximation import fit_bezier_surface_fit_kronecker, BSpline, uniform_knot_bspline_
from curve_utils import DrawSurfs
from fitting_utils import LeastSquares, visualize_weighted_points
from fitting_utils import customsvd
from fitting_utils import remove_outliers
from fitting_utils import standardize_points_torch, sample_points_from_control_points_
from fitting_utils import up_sample_points_in_range
from fitting_utils import up_sample_points_torch_in_range
from guard import guard_sqrt
from utils import draw_geometries, visualize_point_cloud
from utils import rotation_matrix_a_to_b, get_rotation_matrix

from circle_fit_utils import circle_segmentation

draw_surf = DrawSurfs()
EPS = np.finfo(np.float32).eps
torch.manual_seed(2)
np.random.seed(2)
draw_surf = DrawSurfs()
regular_parameters = draw_surf.regular_parameterization(30, 30)


def print_norm(x):
    print("printing norm 2", torch.norm(x))


# ======== pipeline 生成open spline
def forward_pass_open_spline(
        input_points_, control_decoder, nu, nv, viz=False, weights=None, if_optimize=True
):
    nu = nu.cuda(input_points_.get_device())
    nv = nv.cuda(input_points_.get_device())
    with torch.no_grad():
        points_, scales, means, RS = standardize_points_torch(input_points_, weights)

    batch_size = points_.shape[0]
    if viz:
        reg_points = np.copy(points_[:, 0:400])

    # points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
    points = points_.permute(0, 2, 1)
    output = control_decoder(points, weights.T)

    # Chamfer Distance loss, between predicted and GT surfaces
    reconstructed_points = sample_points_from_control_points_(
        nu, nv, output, batch_size
    )
    output = output.view(1, 400, 3)

    out_recon_points = []
    new_outputs = []
    for b in range(batch_size):
        # re-alinging back to original orientation for better comparison
        s = scales[b]

        temp = reconstructed_points[b].clone() * s.reshape((1, 3))
        new_points = torch.inverse(RS[b]) @ torch.transpose(temp, 1, 0)
        temp = torch.transpose(new_points, 1, 0)
        temp = temp + means[b]

        out_recon_points.append(temp)

        temp = output[b] * s.reshape((1, 3))
        temp = torch.inverse(RS[b]) @ torch.transpose(temp, 1, 0)
        temp = torch.transpose(temp, 1, 0)
        temp = temp + means[b]
        new_outputs.append(temp)
        if viz:
            new_points = np.linalg.inv(RS[b]) @ reg_points[b].T
            reg_points[b] = new_points.T
            pred_mesh = tessalate_points(reconstructed_points[b], 30, 30)
            gt_mesh = tessalate_points(reg_points[b], 20, 20)
            draw_geometries([pred_mesh, gt_mesh])

    output = torch.stack(new_outputs, 0)
    reconstructed_points = torch.stack(out_recon_points, 0)
    if if_optimize:
        reconstructed_points = optimize_open_spline_kronecker(reconstructed_points, input_points_, output, deform=True)
    return reconstructed_points, reconstructed_points


def initialize_open_spline_model(modelname, mode):
    from model import DGCNNControlPoints

    control_decoder_ = DGCNNControlPoints(20, num_points=10, mode=mode)
    control_decoder = torch.nn.DataParallel(control_decoder_)
    control_decoder.load_state_dict(
        torch.load(modelname)
    )

    if torch.cuda.device_count() > 1:
        control_decoder_.cuda(1)
    else:
        control_decoder_.cuda(0)
    control_decoder_.eval()
    return control_decoder_


def optimize_close_spline(reconstructed_points, input_points_):
    """
    Assuming that initial point cloud size is greater than or equal to
    400.
    """
    out = reconstructed_points[0]
    out = out.data.cpu().numpy()
    out = out.reshape((31, 30, 3))
    out = out[np.arange(0, 31, 1.5).astype(np.int32)][
          :, np.arange(0, 30, 1.5).astype(np.int32).tolist()
          ]
    out = out.reshape((20 * 21, 3))

    input = input_points_[0]
    N = input.shape[0]
    input = up_sample_points_torch_in_range(input, 2000, 2100)
    # L = np.random.choice(np.arange(N), 30 * 31, replace=False)
    input = input.data.cpu().numpy()

    dist = np.linalg.norm(
        np.expand_dims(out, 1) - np.expand_dims(input, 0), axis=2
    )

    rids, cids = solve_dense(dist)
    matched = input[cids]
    size_u = 21
    size_v = 20
    degree_u = 3
    degree_v = 3

    # Do global surface approximation
    surf = geomdl_fitting.approximate_surface(
        matched.tolist(),
        size_u,
        size_v,
        degree_u,
        degree_v,
        ctrlpts_size_u=10,
        ctrlpts_size_v=10,
    )

    regular_parameters = draw_surf.regular_parameterization(31, 30)
    optimized_points = surf.evaluate_list(regular_parameters)
    optimized_points = torch.from_numpy(np.array(optimized_points).astype(np.float32)).cuda()
    optimized_points = torch.unsqueeze(optimized_points, 0)
    return optimized_points


def optimize_close_spline_kronecker(reconstructed_points,
                                    input_points_,
                                    control_points,
                                    new_cp_size=10,
                                    new_degree=3,
                                    deform=True):
    """
    Assuming that initial point cloud size is greater than or equal to
    400.
    """
    if deform:
        from fitting_optimization import Arap
        arap = Arap()
        new_mesh = arap.deform(reconstructed_points[0].data.cpu().numpy(),
                               input_points_[0].data.cpu().numpy(), viz=False)
        reconstructed_points = torch.from_numpy(np.array(new_mesh.vertices)).cuda()
        reconstructed_points = torch.unsqueeze(reconstructed_points, 0)

    bspline = BSpline()
    N = input_points_.shape[1]
    control_points = control_points[0].data.cpu().numpy()

    new_cp_size = new_cp_size
    new_degree = new_degree

    # Note that boundary parameterization is necessary for the fitting
    parameters = draw_surf.boundary_parameterization(30)
    parameters = np.concatenate([np.random.random((1600 - parameters.shape[0], 2)), parameters], 0)

    _, _, ku, kv = uniform_knot_bspline_(21, 20, 3, 3, 2)

    spline_surf = bspline.create_geomdl_surface(control_points.reshape((21, 20, 3)),
                                                np.array(ku),
                                                np.array(kv),
                                                3, 3)

    # these are randomly sampled points on the surface of the predicted spline
    points = np.array(spline_surf.evaluate_list(parameters))

    input = up_sample_points_torch_in_range(input_points_[0], 2000, 2100)
    input = input.data.cpu().numpy()

    dist = np.linalg.norm(
        np.expand_dims(points, 1) - np.expand_dims(input, 0), axis=2
    )

    rids, cids = solve_dense(dist)
    matched = input[cids]

    _, _, ku, kv = uniform_knot_bspline_(new_cp_size, new_cp_size, new_degree, new_degree, 2)

    NU = []
    NV = []
    for index in range(parameters.shape[0]):
        nu, nv = bspline.basis_functions(parameters[index], new_cp_size, new_cp_size, ku, kv, new_degree, new_degree)
        NU.append(nu)
        NV.append(nv)
    NU = np.concatenate(NU, 1).T
    NV = np.concatenate(NV, 1).T

    new_control_points = fit_bezier_surface_fit_kronecker(matched, NU, NV)
    new_spline_surf = bspline.create_geomdl_surface(new_control_points,
                                                    np.array(ku),
                                                    np.array(kv),
                                                    new_degree, new_degree)

    regular_parameters = draw_surf.regular_parameterization(30, 30)
    optimized_points = new_spline_surf.evaluate_list(regular_parameters)
    optimized_points = torch.from_numpy(np.array(optimized_points).astype(np.float32)).cuda()
    optimized_points = optimized_points.reshape((30, 30, 3))
    optimized_points = torch.cat([optimized_points, optimized_points[0:1]], 0)
    optimized_points = optimized_points.reshape((930, 3))
    optimized_points = torch.unsqueeze(optimized_points, 0)
    return optimized_points


def optimize_open_spline_kronecker(reconstructed_points, input_points_, control_points, new_cp_size=10, new_degree=2,
                                   deform=False):
    """
    Assuming that initial point cloud size is greater than or equal to
    400.
    """
    from fitting_optimization import Arap
    bspline = BSpline()
    N = input_points_.shape[1]
    control_points = control_points[0].data.cpu().numpy()
    if deform:
        arap = Arap(30, 30)
        new_mesh = arap.deform(reconstructed_points[0].data.cpu().numpy(),
                               input_points_[0].data.cpu().numpy(), viz=False)
        reconstructed_points = torch.from_numpy(np.array(new_mesh.vertices)).cuda()
        reconstructed_points = torch.unsqueeze(reconstructed_points, 0)

    new_cp_size = new_cp_size
    new_degree = new_degree

    # Note that boundary parameterization is necessary for the fitting
    # otherwise you 
    parameters = draw_surf.boundary_parameterization(20)
    parameters = np.concatenate([np.random.random((1600 - parameters.shape[0], 2)), parameters], 0)

    _, _, ku, kv = uniform_knot_bspline_(20, 20, 3, 3, 2)
    spline_surf = bspline.create_geomdl_surface(control_points.reshape((20, 20, 3)),
                                                np.array(ku),
                                                np.array(kv),
                                                3, 3)

    # these are randomly sampled points on the surface of the predicted spline
    points = np.array(spline_surf.evaluate_list(parameters))

    input = up_sample_points_torch_in_range(input_points_[0], 1600, 2000)

    L = np.random.choice(np.arange(input.shape[0]), 1600, replace=False)
    input = input[L].data.cpu().numpy()

    dist = np.linalg.norm(
        np.expand_dims(points, 1) - np.expand_dims(input, 0), axis=2
    )

    rids, cids = solve_dense(dist)
    matched = input[cids]

    _, _, ku, kv = uniform_knot_bspline_(new_cp_size, new_cp_size, new_degree, new_degree, 2)

    NU = []
    NV = []
    for index in range(parameters.shape[0]):
        nu, nv = bspline.basis_functions(parameters[index], new_cp_size, new_cp_size, ku, kv, new_degree, new_degree)
        NU.append(nu)
        NV.append(nv)
    NU = np.concatenate(NU, 1).T
    NV = np.concatenate(NV, 1).T

    new_control_points = fit_bezier_surface_fit_kronecker(matched, NU, NV)
    new_spline_surf = bspline.create_geomdl_surface(new_control_points,
                                                    np.array(ku),
                                                    np.array(kv),
                                                    new_degree, new_degree)

    regular_parameters = draw_surf.regular_parameterization(30, 30)
    optimized_points = new_spline_surf.evaluate_list(regular_parameters)
    optimized_points = torch.from_numpy(np.array(optimized_points).astype(np.float32)).cuda()
    optimized_points = torch.unsqueeze(optimized_points, 0)
    return optimized_points



def optimize_open_spline(reconstructed_points, input_points_):
    """
    Assuming that initial point cloud size is greater than or equal to
    400.
    """
    out = reconstructed_points[0]
    out = out.data.cpu().numpy()
    out = out.reshape((30, 30, 3))
    out = out.reshape((900, 3))

    input = input_points_[0]
    N = input.shape[0]
    input = up_sample_points_torch_in_range(input, 1200, 1300)
    input = input.data.cpu().numpy()

    dist = np.linalg.norm(
        np.expand_dims(out, 1) - np.expand_dims(input, 0), axis=2
    )

    rids, cids = solve_dense(dist)
    matched = input[cids]
    size_u = 30
    size_v = 30
    degree_u = 2
    degree_v = 2

    # Do global surface approximation
    try:
        surf = geomdl_fitting.approximate_surface(
            matched.tolist(),
            size_u,
            size_v,
            degree_u,
            degree_v,
            ctrlpts_size_u=10,
            ctrlpts_size_v=10,
        )
    except:
        print("open spline, smaller than 400")
        return reconstructed_points

    regular_parameters = draw_surf.regular_parameterization(30, 30)
    optimized_points = surf.evaluate_list(regular_parameters)
    optimized_points = torch.from_numpy(np.array(optimized_points).astype(np.float32)).cuda()
    optimized_points = torch.unsqueeze(optimized_points, 0)
    return optimized_points


def forward_closed_splines(input_points_, control_decoder, nu, nv, viz=False, weights=None, if_optimize=True):
    batch_size = input_points_.shape[0]
    nu = nu.cuda(input_points_.get_device())  # [30 x 20]
    nv = nv.cuda(input_points_.get_device())  # [30 x 20]

    with torch.no_grad():
        points_, scales, means, RS = standardize_points_torch(input_points_, weights)

    if viz:
        reg_points = points_[:, 0:400]

    # points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
    points = points_.permute(0, 2, 1)  # [B, 1800, 3]
    output = control_decoder(points, weights.T)  # [B, 400, 3]  预测了 (20 x 20) grid 的控制点

    # Chamfer Distance loss, between predicted and GT surfaces
    reconstructed_points = sample_points_from_control_points_(
        nu, nv, output, batch_size
    )  # [B, 900, 3 ]

    closed_reconst = []
    closed_control_points = []

    for b in range(batch_size):
        s = scales[b]
        temp = output[b] * s.reshape((1, 3))
        temp = torch.inverse(RS[b]) @ torch.transpose(temp, 1, 0)
        temp = torch.transpose(temp, 1, 0)
        temp = temp + means[b]

        temp = temp.reshape((20, 20, 3))
        temp = torch.cat([temp, temp[0:1]], 0)
        closed_control_points.append(temp)  # [21, 20, 3]

        temp = (
                reconstructed_points[b].clone() * scales[b].reshape(1, 3)
        )
        temp = torch.inverse(RS[b]) @ temp.T
        temp = torch.transpose(temp, 1, 0) + means[b]
        temp = temp.reshape((30, 30, 3))
        temp = torch.cat([temp, temp[0:1]], 0)
        closed_reconst.append(temp)

    output = torch.stack(closed_control_points, 0)  # [B, 21, 20, 3]
    reconstructed_points = torch.stack(closed_reconst, 0)  # [B, 31, 30, 3]
    reconstructed_points = reconstructed_points.reshape((1, 930, 3))

    if if_optimize and (input_points_.shape[1] > 200):
        reconstructed_points = optimize_close_spline_kronecker(reconstructed_points, input_points_, output)
        reconstructed_points = reconstructed_points.reshape((1, 930, 3))
    return reconstructed_points, None, reconstructed_points


def initialize_closed_spline_model(modelname, mode):
    from model import DGCNNControlPoints

    control_decoder_ = DGCNNControlPoints(20, num_points=10, mode=mode)
    control_decoder = torch.nn.DataParallel(control_decoder_)
    control_decoder.load_state_dict(
        torch.load(modelname)
    )

    if torch.cuda.device_count() > 1:
        control_decoder_.cuda(1)
    else:
        control_decoder_.cuda(0)

    control_decoder_.eval()
    return control_decoder_


class Fit:
    def __init__(self):
        """
        Defines fitting and sampling modules for geometric primitives.
        """
        LS = LeastSquares()
        self.lstsq = LS.lstsq
        self.parameters = {}

    def sample_torus(self, r_major, r_minor, center, axis):
        d_theta = 60
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta

        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1) * r_minor

        circle = np.concatenate([np.zeros((circle.shape[0], 1)), circle], 1)
        circle[:, 1] += r_major

        d_theta = 100
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta
        theta = np.concatenate([theta, np.zeros(1)])

        torus = []
        for i in range(d_theta):
            R = get_rotation_matrix(theta[i])
            torus.append((R @ circle.T).T)

        torus = np.concatenate(torus, 0)
        R = rotation_matrix_a_to_b(np.array([0, 0, 1.0]), axis)
        torus = (R @ torus.T).T
        torus = torus + center
        return torus

    def sample_plane(self, d, n, mean):
        regular_parameters = draw_surf.regular_parameterization(120, 120)
        n = n.reshape(3)
        r1 = np.random.random()
        r2 = np.random.random()
        a = (d - r1 * n[1] - r2 * n[2]) / (n[0] + EPS)
        x = np.array([a, r1, r2]) - d * n

        x = x / np.linalg.norm(x)
        n = n.reshape((1, 3))

        # let us find the perpendicular vector to a lying on the plane
        y = np.cross(x, n)
        y = y / np.linalg.norm(y)

        param = 1 - 2 * np.array(regular_parameters)
        param = param * 0.75

        gridded_points = param[:, 0:1] * x + param[:, 1:2] * y
        gridded_points = gridded_points + mean
        return gridded_points

    def sample_cone_trim(self, c, a, theta, points):
        """
        Trims the cone's height based points. Basically we project 
        the points on the axis and retain only the points that are in
        the range.
        """
        bkp_points = points
        c = c.reshape((3))
        a = a.reshape((3))
        norm_a = np.linalg.norm(a)
        a = a / norm_a
        proj = (points - c.reshape(1, 3)) @ a
        proj_max = np.max(proj)
        proj_min = np.min(proj)

        # find one point on the cone
        k = np.dot(c, a)
        x = (k - a[1] - a[2]) / (a[0] + EPS)
        y = 1
        z = 1
        d = np.array([x, y, z])
        p = a * (np.linalg.norm(d)) / (np.sin(theta) + EPS) * np.cos(theta) + d

        # This is a point on the surface
        p = p.reshape((3, 1))

        # Now rotate the vector p around axis a by variable degree
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        points = []
        normals = []
        c = c.reshape((3, 1))
        a = a.reshape((3, 1))
        rel_unit_vector = p - c
        rel_unit_vector = (p - c) / np.linalg.norm(p - c)
        rel_unit_vector_min = rel_unit_vector * (proj_min) / (np.cos(theta) + EPS)
        rel_unit_vector_max = rel_unit_vector * (proj_max) / (np.cos(theta) + EPS)

        for j in range(100):
            # p_ = (p - c) * (0.01) * j
            p_ = rel_unit_vector_min + (rel_unit_vector_max - rel_unit_vector_min) * 0.01 * j

            d_points = []
            d_normals = []
            for d in range(50):
                degrees = 2 * np.pi * 0.01 * d * 2
                R = np.eye(3) + np.sin(degrees) * K + (1 - np.cos(degrees)) * K @ K
                rotate_point = R @ p_
                d_points.append(rotate_point + c)
                d_normals.append(rotate_point - np.linalg.norm(rotate_point) / np.cos(theta) * a / norm_a)

            # repeat the points to close the circle
            d_points.append(d_points[0])
            d_normals.append(d_normals[0])

            points += d_points
            normals += d_normals

        points = np.stack(points, 0)[:, :, 0]
        normals = np.stack(normals, 0)[:, :, 0]
        normals = normals / (np.expand_dims(np.linalg.norm(normals, axis=1), 1) + EPS)

        # projecting points to the axis to trim the cone along the height.
        proj = (points - c.reshape((1, 3))) @ a
        proj = proj[:, 0]
        indices = np.logical_and(proj < proj_max, proj > proj_min)
        # project points on the axis, remove points that are beyond the limits.
        return points[indices], normals[indices]

    def sample_cone(self, c, a, theta):
        norm_a = np.linalg.norm(a)
        a = a / norm_a

        # find one point on the cone
        k = np.dot(c, a)
        x = (k - a[1] - a[2]) / (a[0] + EPS)
        y = 1
        z = 1
        d = np.array([x, y, z])
        p = a * (np.linalg.norm(d)) / (np.sin(theta) + EPS) * np.cos(theta) + d

        # This is a point on the surface
        p = p.reshape((3, 1))

        # Now rotate the vector p around axis a by variable degree
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        points = []
        normals = []
        c = c.reshape((3, 1))
        a = a.reshape((3, 1))
        for j in range(100):
            p_ = (p - c) * (0.01) * j
            d_points = []
            d_normals = []
            for d in range(50):
                degrees = 2 * np.pi * 0.01 * d * 2
                R = np.eye(3) + np.sin(degrees) * K + (1 - np.cos(degrees)) * K @ K
                rotate_point = R @ p_
                d_points.append(rotate_point + c)
                d_normals.append(rotate_point - np.linalg.norm(rotate_point) / np.cos(theta) * a / norm_a)
            # repeat the points to close the circle
            d_points.append(d_points[0])
            d_normals.append(d_normals[0])

            points += d_points
            normals += d_normals

        points = np.stack(points, 0)[:, :, 0]
        normals = np.stack(normals, 0)[:, :, 0]
        normals = normals / (np.expand_dims(np.linalg.norm(normals, axis=1), 1) + EPS)

        points = points - c.reshape((1, 3))
        points = 2 * points / (np.max(np.linalg.norm(points, ord=2, axis=1, keepdims=True)) + EPS)
        points = points + c.reshape((1, 3))
        return points, normals

    def sample_sphere(self, radius, center, N=1000):

        theta = 1 - 2 * np.random.random(N) * 3.14
        phi = 1 - 2 * np.random.random(N) * 3.14
        points = np.stack([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi),
                           np.sin(theta)], 1)
        normals = points
        points = points * radius

        points = points + center
        return points, normals

    def sample_sphere_v2(self, radius, center, N=1000):
        center = center.reshape((1, 3))
        d_theta = 100
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta
        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1)
        lam = np.linspace(-1 + 1e-7, 1 - 1e-7, 100)
        radii = radius * np.sqrt(1 - lam ** 2)
        circle = np.concatenate([circle] * lam.shape[0], 0)
        spread_radii = np.repeat(radii, d_theta, 0)
        new_circle = circle * spread_radii.reshape((-1, 1))
        height = np.repeat(lam, d_theta, 0)
        points = np.concatenate([new_circle, height.reshape((-1, 1))], 1)
        points = points - np.mean(points, 0)
        normals = points / np.linalg.norm(points, axis=1, keepdims=True)
        points = points + center
        return points, normals

    def sample_cylinder_trim(self, radius, center, axis, points, N=1000):
        """
        :param center: center of size 1 x 3
        :param radius: radius of the cylinder
        :param axis: axis of the cylinder, size 3 x 1
        """
        center = center.reshape((1, 3))
        axis = axis.reshape((3, 1))

        d_theta = 60
        d_height = 100

        R = self.rotation_matrix_a_to_b(np.array([0, 0, 1]), axis[:, 0])

        # project points on to the axis
        points = points - center

        projection = points @ axis
        arg_min_proj = np.argmin(projection)
        arg_max_proj = np.argmax(projection)

        min_proj = np.squeeze(projection[arg_min_proj])
        max_proj = np.squeeze(projection[arg_max_proj])

        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta

        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1)
        circle = np.concatenate([circle] * 2 * d_height, 0) * radius

        normals = np.concatenate([circle, np.zeros((circle.shape[0], 1))], 1)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        height = np.expand_dims(np.linspace(min_proj, max_proj, 2 * d_height), 1)
        height = np.repeat(height, d_theta, axis=0)
        try:
            points = np.concatenate([circle, height], 1)
        except:
            import ipdb
            ipdb.set_trace()
        points = R @ points.T
        points = points.T + center
        normals = (R @ normals.T).T

        return points, normals

    def sample_cylinder(self, radius, center, axis, N=1000):
        """
        :param center: center of size 1 x 3
        :param radius: radius of the cylinder
        :param axis: axis of the cylinder, size 3 x 1
        """
        d_theta = 30
        d_height = 200
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta

        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1)
        circle = np.concatenate([circle] * 2 * d_height, 0) * radius

        heights = [np.ones(d_theta) * i for i in range(-d_height, d_height)]
        heights = np.concatenate(heights, 0).reshape((-1, 1)) * 0.01

        points = np.concatenate([circle, heights], 1)
        N = points.shape[0]

        normals = np.concatenate([circle, np.zeros((N, 1))], 1)
        normals = normals / np.linalg.norm(normals, axis=1).reshape((N, 1))
        R = self.rotation_matrix_a_to_b(np.array([0, 0, 1]), axis)

        points = R @ points.T
        points = points.T + center
        normals = (R @ normals.T).T

        return points, normals

    def fit_plane_numpy(self, points, normals, weights):
        """
        Fits plane
        :param points: points with size N x 3
        :param weights: weights with size N x 1
        """
        X = points - np.sum(weights * points, 0).reshape((1, 3)) / np.sum(weights, 0)
        _, s, V = np.linalg.svd(weights * X, compute_uv=True)
        a = V.T[:, np.argmin(s)]
        a = np.reshape(a, (1, 3))
        d = np.sum(weights * (a @ points.T).T) / np.sum(weights, 0)
        return a, d

    def fit_plane_torch(self, points, normals, weights, ids=0, show_warning=False, nofilter=False, filter_ratio=0.5):
        """
        Fits plane
        :param points: points with size N x 3
        :param weights: weights with size N x 1
        """
        if not nofilter:
            center = points.mean(0).reshape((1, 3))
            index = torch.argsort((points - center).pow(2).sum(-1))  # 递增
            index = index[:int(index.shape[0] * filter_ratio)]
            points = points[index, :]
            normals = normals[index, :]
            weights = weights[index, :]

        weights_sum = torch.sum(weights) + EPS

        X = points - torch.sum(weights * points, 0).reshape((1, 3)) / weights_sum

        weighted_X = weights * X
        np_weighted_X = weighted_X.data.cpu().numpy()
        if np.linalg.cond(np_weighted_X) > 1e5:
            if show_warning:
                print("condition number is large in plane!", np.sum(np_weighted_X))
                print(torch.sum(points), torch.sum(weights))

        U, s, V = customsvd(weighted_X)
        a = V[:, -1]
        a = torch.reshape(a, (1, 3))

        # ==================================================================================== 1336_DoorKnob_clean
        # a[0, 0] = a[0, 1] = 0

        d = torch.sum(weights * (a @ points.permute(1, 0)).permute(1, 0)) / weights_sum

        '''
        center = points.mean(0).reshape((1, 3))
        index = torch.argsort((points - center).pow(2).sum(-1))  # 递增
        index = index[:index.shape[0] // 2]
        points = points[index, :]
        normals = normals[index, :]
        a = normals.mean(0).reshape((1, 3))
        d = torch.mean(a @ points.transpose(1, 0))
        '''
        return a, d

    def fit_sphere_numpy(self, points, normals, weights):
        dimension = points.shape[1]
        N = weights.shape[0]
        sum_weights = np.sum(weights)
        A = 2 * (- points + np.sum(points * weights, 0) / sum_weights)
        dot_points = np.sum(points * points, 1)
        normalization = np.sum(dot_points * weights) / sum_weights
        Y = dot_points - normalization
        Y = Y.reshape((N, 1))
        A = weights * A
        Y = weights * Y
        center = -np.linalg.lstsq(A, Y)[0].reshape((1, dimension))
        radius = np.sqrt(np.sum(weights[:, 0] * np.sum((points - center) ** 2, 1)) / sum_weights)
        return center, radius

    def fit_sphere_torch(self, points, normals, weights, ids=0, show_warning=False):

        N = weights.shape[0]
        sum_weights = torch.sum(weights) + EPS
        A = 2 * (- points + torch.sum(points * weights, 0) / sum_weights)

        dot_points = weights * torch.sum(points * points, 1, keepdim=True)

        normalization = torch.sum(dot_points) / sum_weights

        Y = dot_points - normalization
        Y = Y.reshape((N, 1))
        A = weights * A
        Y = weights * Y

        if np.linalg.cond(A.data.cpu().numpy()) > 1e8:
            if show_warning:
                print("condition number is large in sphere!")

        center = -self.lstsq(A, Y, 0.01).reshape((1, 3))
        radius_square = torch.sum(weights[:, 0] * torch.sum((points - center) ** 2, 1)) / sum_weights
        radius_square = torch.clamp(radius_square, min=1e-3)
        radius = guard_sqrt(radius_square)
        return center, radius

    def fit_cylinder_numpy(self, points, normals, weights):
        _, s, V = np.linalg.svd(weights * normals, compute_uv=True)
        a = V.T[:, np.argmin(s)]
        a = np.reshape(a, (1, 3))

        # find the projection onto a plane perpendicular to the axis
        a = a.reshape((3, 1))
        a = a / (np.linalg.norm(a, ord=2) + EPS)

        prj_circle = points - ((points @ a).T * a).T
        center, radius = self.fit_sphere_numpy(prj_circle, normals, weights)
        return a, center, radius

    def fit_cylinder_torch(self, points, normals, weights, ids=0, show_warning=False):
        # compute
        # U, s, V = torch.svd(weights * normals)

        points, normals, weights = points.float(), normals.float(), weights.float()

        weighted_normals = weights * normals

        if np.linalg.cond(weighted_normals.data.cpu().numpy()) > 1e5:
            if show_warning:
                print("condition number is large in cylinder")
                print(torch.sum(normals).item(), torch.sum(points).item(), torch.sum(weights).item())

        '''
        if weighted_normals.shape[0] > 600:
            centers = points.mean(0)
            dist = (points - centers).pow(2).sum(dim=-1)  # n x 1
            min_dist_index = torch.argsort(dist)[:weighted_normals.shape[0] // 3]
            weighted_normals = weighted_normals[min_dist_index, :]
            points = points[min_dist_index, :]'''

        # print(points.shape)
        # visualize_point_cloud(points, viz=True)

        U, s, V = torch.svd(weighted_normals)
        a = V[:, -1]
        '''
        a, _ = torch.lstsq(torch.zeros((points.shape[0], 1)).float().cuda(), A=weighted_normals)
        a = a[:3].reshape((3, 1))
        '''
        # find the projection onto a plane perpendicular to the axis

        a = a.reshape((3, 1))

        # =============================================================================================== 1336_DoorKnob_clean
        # a[0, 0] = a[1, 0] = 0

        a = a / (torch.norm(a, 2) + EPS)



        prj_circle = points - ((points @ a).permute(1, 0) * a).permute(1, 0)

        # visualize_point_cloud(prj_circle, viz=True)

        circle, center, radius = circle_segmentation(prj_circle.cpu().numpy())

        # =============================================================================================== 1336_DoorKnob_clean
        # center[0] = center[1] = 0

        return a, torch.from_numpy(center), radius

    def fit_cone_torch(self, points, normals, weights, ids=0, show_warning=False):
        """ Need to incorporate the cholesky decomposition based
        least square fitting because it is stable and faster."""


        # center = points.mean(0).reshape((1, 3))
        # index = torch.argsort((points - center).pow(2).sum(-1))  # 递增
        # index = index[:index.shape[0] // 2]
        # points = points[index, :]
        # normals = normals[index, :]
        # weights = weights[index, :]

        # visualize_point_cloud(points, viz=True)

        '''
        N = points.shape[0]
        A = weights * normals
        Y = torch.sum(normals * points, 1).reshape((N, 1))
        Y = weights * Y
        '''
        '''
        # if condition number is too large, return a very zero cone.
        if np.linalg.cond(A.data.cpu().numpy()) > 1e5:
            if show_warning:
                print("condition number is large, cone")
                print(torch.sum(normals).item(), torch.sum(points).item(), torch.sum(weights).item())
            return torch.zeros((1, 3)).cuda(points.get_device()), torch.Tensor([[1.0, 0.0, 0.0]]).cuda(
                points.get_device()), torch.zeros(1).cuda(points.get_device())
        '''
        N = points.shape[0]
        Y = torch.sum(normals * points, 1).reshape((N, 1))
        c, _ = torch.lstsq(Y, A=normals)
        c = c[:3].reshape((3, 1))

        # ========================================================================================== for 1336_DoorKnob_clean
        # c[0, 0] = c[1, 0] = 0


        a, _ = self.fit_plane_torch(points, normals, weights, nofilter=True)  # 1 x 3

        if ((c.reshape((3,)) - points[0]) * a.reshape((3,))).sum() < 0:
            a = -a

        '''
        if torch.sum(normals @ a.transpose(1, 0)) > 0:
            # we want normals to be pointing outside and axis to
            # be pointing inside the cone.
            a = - 1 * a
        '''

        # ====
        # trick LS
        # ====
        for i in range(3):
            if torch.abs(a[0, i]) >= 0.98:
                tmp = torch.zeros_like(a)
                tmp[0, i] = 1 if a[0, i] > 0 else -1
                a = tmp
            if torch.abs(c[i, 0]) <= 0.1:
                c[i, 0] = 0

        diff = points - c.transpose(1, 0)
        diff = torch.nn.functional.normalize(diff, p=2, dim=1)
        diff = diff @ a.transpose(1, 0)

        # This is done to avoid the numerical issue when diff = 1 or -1
        # the derivative of acos becomes inf
        diff = torch.abs(diff)
        diff = torch.clamp(diff, max=0.999)
        theta = torch.sum(weights * torch.acos(diff)) / (torch.sum(weights) + EPS)
        theta = torch.clamp(theta, min=1e-3, max=3.142 / 2 - 1e-3)  # 0 - 90
        return c, a, theta

    def rotation_matrix_a_to_b(self, A, B):
        """
        Finds rotation matrix from vector A in 3d to vector B
        in 3d.
        B = R @ A
        """
        cos = np.dot(A, B)
        sin = np.linalg.norm(np.cross(B, A))
        u = A
        v = B - np.dot(A, B) * A
        v = v / (np.linalg.norm(v) + EPS)
        w = np.cross(B, A)
        w = w / (np.linalg.norm(w) + EPS)
        F = np.stack([u, v, w], 1)
        G = np.array([[cos, -sin, 0],
                      [sin, cos, 0],
                      [0, 0, 1]])
        # B = R @ A
        try:
            R = F @ G @ np.linalg.inv(F)
        except:
            R = np.eye(3, dtype=np.float32)
        return R

    def reg_lstsq(self, A, y, lamb=0):
        n_col = A.shape[1]
        return np.linalg.lstsq(A.T.dot(A) + lamb * np.identity(n_col), A.T.dot(y))

    def reg_lstsq_torch(self, A, y, lamb=0):
        n_col = A.shape[1]
        A_dash = A.permute(1, 0) @ A + lamb * torch.eye(n_col)
        y_dash = A.permute(1, 0) @ y

        center = self.lstsq(y_dash, A_dash)
        return center


# 非 pipe line 方法，ignore！
def fit_one_shape(data, fitter):
    """
    Fits primitives/splines to one shape
    """
    input_shape = []
    reconstructed_shape = []
    fitter.fitting.parameters = {}
    gt_points = {}

    for part_index, d in enumerate(data):
        points, normals, labels, _ = d
        weights = np.ones((points.shape[0], 1), dtype=np.float32)
        if labels[0] in [0, 9, 6, 7]:
            # closed bspline surface
            # Ignore the patches that are very small, that is smaller than 1%.
            if points.shape[0] < 100:
                continue
            recon_points = fitter.forward_pass_closed_spline(points, weights=weights, ids=part_index)

        elif labels[0] == 1:
            # Fit plane
            recon_points = fitter.forward_pass_plane(points, normals, weights, ids=part_index)

        elif labels[0] == 3:
            # Cone
            recon_points = fitter.forward_pass_cone(points, normals, weights, ids=part_index)

        elif labels[0] == 4:
            # cylinder
            recon_points = fitter.forward_pass_cylinder(points, normals, weights, ids=part_index)

        elif labels[0] == 5:
            # sphere
            recon_points = fitter.forward_pass_sphere(points, normals, weights, ids=part_index)

        elif labels[0] in [2, 8]:
            # open splines
            recon_points = fitter.forward_pass_open_spline(points, ids=part_index)
        gt_points[part_index] = points
        reconstructed_shape.append(recon_points)
    return gt_points, reconstructed_shape


#  非 pipe line 方法，ignore！
def fit_one_shape_torch(data, fitter, weights, bw, eval=False, sample_points=False, if_optimize=False,
                        if_visualize=False):
    """
    Fits primitives/splines to 
    """
    input_shape = []
    reconstructed_shape = []
    fitter.fitting.parameters = {}
    gt_points = {}
    spline_count = 0

    for _, d in enumerate(data):
        points, normals, labels, gpoints, segment_indices, part_index = d
        # NOTE: part index and label index are different when one of the predicted
        # labels are missing.
        part_index, label_index = part_index
        N = points.shape[0]
        if not eval:
            weight = weights[:, part_index:part_index + 1] + EPS
            drop_indices = torch.arange(0, N, 2)
            points = points[drop_indices]
            normals = normals[drop_indices]
            weight = weight[drop_indices]

        else:
            weight = weights[segment_indices, part_index:part_index + 1] + EPS

        if not eval:
            # in the training mode, only process upto 5 splines
            # because of the memory constraints.
            if labels in [0, 2, 6, 7, 9, 8]:
                spline_count += 1
                if spline_count > 4:
                    reconstructed_shape.append(None)
                    gt_points[label_index] = None
                    fitter.fitting.parameters[label_index] = None
                    continue
            else:
                # down sample points for geometric primitives, what is the point
                N = points.shape[0]
                drop_indices = torch.arange(0, N, 2)
                points = points[drop_indices]
                normals = normals[drop_indices]
                weight = weight[drop_indices]

        if points.shape[0] < 20:
            reconstructed_shape.append(None)
            gt_points[label_index] = None
            fitter.fitting.parameters[label_index] = None
            continue

        if labels in [0, 9, 6, 7]:
            # closed bspline surface

            if points.shape[0] < 100:
                # drop smaller patches
                reconstructed_shape.append(None)
                gt_points[label_index] = None
                fitter.fitting.parameters[label_index] = None
                continue

            if eval:
                # since this is a eval mode, weights are all one.
                Z = points.shape[0]
                points = torch.from_numpy(remove_outliers(points.data.cpu().numpy()).astype(np.float32)).cuda(
                    points.get_device())
                weight = weight[0:points.shape[0]]

                # Note: we can apply poisson disk sampling to remove points.
                # Rarely results in removal of points.
                points, weight = up_sample_points_in_range(points, weight, 1400, 1800)

            recon_points = fitter.forward_pass_closed_spline(points, weights=weight, ids=label_index,
                                                             if_optimize=if_optimize and (Z > 200))

        elif labels == 1:
            # Fit plane
            recon_points = fitter.forward_pass_plane(points, normals, weight, ids=label_index,
                                                     sample_points=sample_points)

        elif labels == 3:
            # Cone
            recon_points = fitter.forward_pass_cone(points, normals, weight, ids=label_index,
                                                    sample_points=sample_points)

        elif labels == 4:
            # cylinder
            recon_points = fitter.forward_pass_cylinder(points, normals, weight, ids=label_index,
                                                        sample_points=sample_points)

        elif labels == 5:
            # "sphere"
            recon_points = fitter.forward_pass_sphere(points, normals, weight, ids=label_index,
                                                      sample_points=sample_points)

        elif labels in [2, 8]:
            # open splines
            if points.shape[0] < 100:
                reconstructed_shape.append(None)
                gt_points[label_index] = None
                fitter.fitting.parameters[label_index] = None
                continue
            if eval:
                # in the eval mode, make the number of points per segment to lie in a range that is suitable for the spline network.
                # remove outliers. Only occur rarely, but worth removing them.
                points = torch.from_numpy(remove_outliers(points.data.cpu().numpy()).astype(np.float32)).cuda(
                    points.get_device())
                weight = weight[0:points.shape[0]]
                points, weight = up_sample_points_in_range(points, weight, 1000, 1500)

            recon_points = fitter.forward_pass_open_spline(points, weights=weight, ids=label_index,
                                                           if_optimize=if_optimize)

        if if_visualize:
            try:
                gt_points[label_index] = torch.from_numpy(gpoints).cuda(points.get_device())
            except:
                gt_points[label_index] = None
        else:
            gt_points[label_index] = gpoints

        reconstructed_shape.append(recon_points)
    return gt_points, reconstructed_shape



# ================================
"""
By Liu Shun
"""
# TODO
# pipeline使用的方法
def my_fit_one_shape(data, fitter, num_thresh=10, bad_insts=None, fn_id="0", save_para=False):
    """
    Fits primitives/splines to one shape
    """
    reconstructed_shape = []
    fitter.fitting.parameters = {}
    gt_points = {}

    # todo
    # 优化 spline 的生成
    if_optimize = True

    for part_index, d in enumerate(data):
        points, normals, labels = d[:3]

        # visualize_point_cloud(points, viz=True)

        weights = torch.ones((points.shape[0], 1)).cuda()

        if points.shape[0] < num_thresh or (bad_insts is not None and part_index in bad_insts):
            print("{} instance too small".format(part_index))
            fitter.fitting.parameters[part_index] = ["none"]
            recon_points = points.cpu().numpy()

        elif labels == 1:
            # Fit plane
            recon_points = fitter.forward_pass_plane(points, normals, weights, ids=part_index, sample_points=True)

        elif labels == 3:
            # Cone
            recon_points = fitter.forward_pass_cone(points, normals, weights, ids=part_index, sample_points=True)

        elif labels == 2:
            # cylinder
            # visualize_point_cloud(points, viz=True)
            recon_points = fitter.forward_pass_cylinder(points, normals, weights,
                                                        ids=part_index, sample_points=True, RANSAC=False)

        elif labels == 4:
            # sphere
            recon_points = fitter.forward_pass_sphere(points, normals, weights, ids=part_index, sample_points=True)

        elif labels in [0, ]:
            # closed (or opened) b-spline surface

            if points.shape[0] < 100:
                # drop smaller patches
                reconstructed_shape.append(None)
                gt_points[part_index] = None
                fitter.fitting.parameters[part_index] = None
                print("drop small(<100) spline patch id {}".format(part_index))
                continue

            if True:
                # if eval:
                # since this is a eval mode, weights are all one.
                Z = points.shape[0]
                points = torch.from_numpy(remove_outliers(points.data.cpu().numpy()).astype(np.float32)).cuda(
                    points.get_device())
                weight = weights[0:points.shape[0]]

                # Note: we can apply poisson disk sampling to remove points.
                # Rarely results in removal of points.
                points, weight = up_sample_points_in_range(points, weight, 1400, 1800)

            """
            核心函数
            """
            recon_points = fitter.forward_pass_closed_spline(points, weights=weight, ids=part_index,
                                                             if_optimize=if_optimize and (Z > 200))
            recon_points = recon_points[0].cpu().numpy()
            print(recon_points.shape)  # 930 x 3
            temp = recon_points.reshape((31, 30, 3))
            pred_mesh = tessalate_points(temp, 31, 30)
            pred_mesh.paint_uniform_color([1, 0.0, 0])
            open3d.io.write_triangle_mesh(
                "./pred_splines/pred_{}_{}.ply".format(
                    fn_id, part_index
                ),
                pred_mesh,
            )
        gt_points[part_index] = points
        reconstructed_shape.append(recon_points)
    return gt_points, reconstructed_shape


# 非 pipe line 方法，ignore！
def my_fit_one_shape_torch(data, fitter, weights, eval=False, sample_points=False,
                           if_visualize=False):
    """
    Fits primitives/splines to
    """
    input_shape = []
    reconstructed_shape = []
    fitter.fitting.parameters = {}
    gt_points = {}
    spline_count = 0

    for _, d in enumerate(data):
        points, normals, labels, gpoints, segment_indices, part_index = d
        # NOTE: part index and label index are different when one of the predicted
        # labels are missing.
        part_index = labels
        label_index = labels
        N = points.shape[0]
        if not eval:
            weight = weights[:, part_index:part_index + 1] + EPS
            drop_indices = torch.arange(0, N, 2)
            points = points[drop_indices]
            normals = normals[drop_indices]
            weight = weight[drop_indices]

        else:
            weight = weights[segment_indices, part_index:part_index + 1] + EPS
            print(weight.shape)

        if not eval:
            # in the training mode, only process upto 5 splines
            # because of the memory constraints.
            if labels in [0, 2, 6, 7, 9, 8]:
                spline_count += 1
                if spline_count > 4:
                    reconstructed_shape.append(None)
                    gt_points[label_index] = None
                    fitter.fitting.parameters[label_index] = None
                    continue
            else:
                # down sample points for geometric primitives, what is the point
                N = points.shape[0]
                drop_indices = torch.arange(0, N, 2)
                points = points[drop_indices]
                normals = normals[drop_indices]
                weight = weight[drop_indices]

        if points.shape[0] < 20:
            reconstructed_shape.append(None)
            gt_points[label_index] = None
            fitter.fitting.parameters[label_index] = None
            continue

        elif labels == 1:
            # Fit plane
            recon_points = fitter.forward_pass_plane(points, normals, weight, ids=label_index,
                                                     sample_points=sample_points)

        elif labels == 3:
            # Cone
            recon_points = fitter.forward_pass_cone(points, normals, weight, ids=label_index,
                                                    sample_points=sample_points)

        elif labels == 2:
            # cylinder
            recon_points = fitter.forward_pass_cylinder(points, normals, weight, ids=label_index,
                                                        sample_points=sample_points)

        elif labels == 4:
            # "sphere"
            recon_points = fitter.forward_pass_sphere(points, normals, weight, ids=label_index,
                                                      sample_points=sample_points)

        if if_visualize:
            try:
                gt_points[label_index] = torch.from_numpy(gpoints).cuda(points.get_device())
            except:
                gt_points[label_index] = None
        else:
            gt_points[label_index] = gpoints

        reconstructed_shape.append(recon_points)
    return gt_points, reconstructed_shape


from proj_2_edge_utils import plane_plane_inter_line, face_face_inter_map, plane_cylinder_inter_line, \
    points_project_to_line, get_edges_between_insts, plane_cone_inter_line, cylinder_cone_inter_line, vector_cos, \
    line_line_inter, line_circle_inter, fitter_point, get_line_point_d, cylinder_cylinder_inter_line, \
    get_circle_point_theta, get_circle_two_point_theta, bad_points_mask, plane_sphere_inter_line, np_process

if __name__ == "__main__":
    '''    
    primitives = np.loadtxt("./src/None_51_inst.txt").astype(np.int32)
    types = np.loadtxt("./src/None_51_type.txt").astype(np.int32)
    raw_points = np.loadtxt("./src/51_gt_points.txt", delimiter=";")
    print(primitives.shape, types.shape, raw_points.shape)

    from fitting_optimization import MyFittingModule
    fitter = MyFittingModule()

    data = []
    primitive_ids = np.unique(primitives)
    print(primitive_ids)

    for inst in primitive_ids:

        index = primitives == inst  # N,
        points = torch.from_numpy(raw_points[index, :3]).cuda()
        normals = torch.from_numpy(raw_points[index, 3:]).cuda()

        count = np.bincount(types[index])
        labels = int(np.argmax(count))

        part_index = labels

        data.append((points, normals, labels, points, index, part_index))

    weights = torch.ones((raw_points.shape[0], 5)).cuda()

    gt_points, reconstructed_shape = my_fit_one_shape_torch(data, fitter, weights, eval=True,
                        sample_points=True, if_visualize=True)

    for i, data in enumerate(reconstructed_shape):
        print(data.shape)
        np.savetxt("./dst/{}.txt".format(i), data, fmt="%0.4f", delimiter=';')
    '''

    id = "779"
    # id = "00052045" # 2362  2045   0779

    nn_num_thresh = 2

    num_thresh = 10

    fliter_bad_points = False

    plane_sample_ratio = 0.25

    is_gt = True

    _dir = "src_spline"
    save_dir = "dst_spline"

    corner_dis_thresh = 0.01

    if id in ["962", "952", "111", "978"]:
        is_gt = True

    if id in ["962"]:
        corner_dis_thresh = 0.001

    if is_gt:
        '''
        raw = np.loadtxt("./{}/{}_insEdge.txt".format(_dir, id), delimiter=";")
        raw_points = raw[:, :6]
        types = np.argmax(raw[:, 6:11], axis=-1).astype(np.int32)
        primitives = raw[:, 11].astype(np.int32)
        is_edge = np.zeros((raw_points.shape[0], ), dtype=bool)

        np.savetxt("./{}/{}_gt_points.txt".format(_dir, id), raw_points, fmt="%0.4f", delimiter=';')
        '''
        raw_points = np.loadtxt("./{}/{}_gt_points.txt".format(_dir, id), delimiter=";")
        primitives = np.loadtxt("./{}/{}_inst.txt".format(_dir, id)).astype(np.int32)
        types = np.loadtxt("./{}/{}_type.txt".format(_dir, id)).astype(np.int32)
        # edge_confi = np.loadtxt("./{}/{}_gt_edges.txt".format(_dir, id), delimiter=";")
        # is_edge = edge_confi.argmax(-1) == 1  # N x 1

    else:
        primitives = np.loadtxt("./{}/None_{}_inst.txt".format(_dir, id)).astype(np.int32)
        # primitives = np.loadtxt("./{}/{}_inst.txt".format(_dir, id)).astype(np.int32)
        types = np.loadtxt("./{}/None_{}_type.txt".format(_dir, id)).astype(np.int32)
        raw_points = np.loadtxt("./{}/{}_gt_points.txt".format(_dir, id), delimiter=";")
        edge_confi = np.loadtxt("./{}/None_{}_edge_confi.txt".format(_dir, id), delimiter=";")
        is_edge = edge_confi.argmax(-1) == 1  # N x 1

    # print(primitives.shape, types.shape, raw_points.shape)
    # visualize_point_cloud(raw_points, viz=True)

    from fitting_optimization import MyFittingModule

    fitter = MyFittingModule(plane_sample_ratio)

    inst_data = []
    primitive_ids = np.unique(primitives)
    print(primitive_ids)

    strict_edge_idx = get_edges_between_insts(raw_points, primitives).cpu().numpy()  # 得到inst之间的边界点索引

    for inst in primitive_ids:

        index = primitives == inst  # N,

        count = np.bincount(types[index])
        labels = np.argmax(count)

        if labels == 2:  # cylinder
            # index = index & ~is_edge  # 除去边界，使得拟合更稳定
            index = index & ~strict_edge_idx

        if labels == 3:  # Cone
            index = index & ~strict_edge_idx

        points = raw_points[index, :3]
        normals = raw_points[index, 3:]

        inst_data.append((torch.from_numpy(points).cuda(),
                          torch.from_numpy(normals).cuda(),
                          labels, index))

    # parsenet my_fit_one_shape

    bad_insts = []

    if id == "952":
        bad_insts.append(15)

    gt_points, reconstructed_shape = my_fit_one_shape(inst_data, fitter, num_thresh, bad_insts, fn_id=id)

    with open("param_{}.txt".format(id), "w") as para_f:
        for key in fitter.fitting.parameters.keys():
            s = "id {}: ".format(key)
            for item in fitter.fitting.parameters[key]:
                if isinstance(item, torch.Tensor):
                    item = item.flatten().cpu().numpy()
                s += str(item) + " , "
            s += "\n"
            para_f.write(s)

    # fitter
    primitive_ids = torch.from_numpy(primitive_ids)

    # 将点云中不在拟合出来的instance面的点过滤，再求intermap
    if fliter_bad_points:
        _bad_points_mask = bad_points_mask(raw_points[:, :3], primitives, primitive_ids, fitter)
        _mask = (~_bad_points_mask & 1).astype(bool)
        # visualize_point_cloud(raw_points[_mask, :3], viz=True)
        intermap = face_face_inter_map(raw_points[_mask, :3], primitives[_mask], primitive_ids,
                                       nn_num_thresh)  # 30 x 30  bool mat
    else:
        intermap = face_face_inter_map(raw_points[:, :3], primitives, primitive_ids, nn_num_thresh)  # 30 x 30  bool mat

    inter_para_set = {}
    for _id in primitive_ids:
        inter_para_set[int(_id)] = {}

    # 求解instance 与 instance之间的交线
    gen_edges = np.ones((0, 3), dtype=np.float64)
    for i, data in enumerate(reconstructed_shape):
        np.savetxt("./{}/{}_{}_{}{}.txt".format(save_dir, id, i, int(primitive_ids[i]),
                                                 "_close_spline" if fitter.fitting.parameters[i][0] == "closed-spline" else ""),
                   data, fmt="%0.4f", delimiter=';')
        inter_faces = torch.where(intermap[primitive_ids[i], :] == 1)[0]

        if fitter.fitting.parameters[i][0] == "none":
            inst_id = int(primitive_ids[i])
            print("intersect ignore inst {}", inst_id)
            intermap[inst_id, :] = False
            intermap[:, inst_id] = False
            continue

        for _id in range(inter_faces.shape[0]):
            try:
                n_index = int(torch.where(primitive_ids == inter_faces[_id])[0][0])
                inst_id_1 = int(primitive_ids[i])
                inst_id_2 = int(primitive_ids[n_index])
                print("============id {} {}".format(inst_id_1, inst_id_2))
                para1 = fitter.fitting.parameters[i]
                para2 = fitter.fitting.parameters[n_index]
                if para1[0] == para2[0] == 'plane':
                    if inst_id_2 in inter_para_set[inst_id_1].keys():
                        print("==== repeat")
                        continue

                    flag, k, ds = plane_plane_inter_line(para1, para2)

                    if flag == "None":
                        intermap[inst_id_1, inst_id_2] = intermap[inst_id_2, inst_id_1] = False
                        continue

                    print(k, ds)
                    d = ds[0]
                    for _d in ds:
                        if np.abs(_d).max() > 5:
                            continue
                        else:
                            d = _d
                    samples = np.arange(-0.5, 0.5, 0.001).reshape((-1, 1))
                    line_points = d + samples * k.reshape((1, -1))

                    inter_para_set[inst_id_1][inst_id_2] = ["line", k, d]  # K @ [x, y ,z] = d
                    inter_para_set[inst_id_2][inst_id_1] = ["line", k, d]

                    gen_edges = np.concatenate((gen_edges, line_points), axis=0)

                elif para1[0] == 'plane' and para2[0] == "cylinder":
                    gen_paras = plane_cylinder_inter_line(para1, para2)
                    if gen_paras[0] == "None":
                        intermap[inst_id_1, inst_id_2] = intermap[inst_id_2, inst_id_1] = False
                        continue

                    elif gen_paras[0] == "line":
                        inter_para_set[inst_id_1][inst_id_2] = ["line", gen_paras[1],
                                                                gen_paras[2].reshape((3,))]  # K @ [x, y ,z] = d
                        inter_para_set[inst_id_2][inst_id_1] = ["line", gen_paras[1], gen_paras[2].reshape((3,))]

                    elif gen_paras[0] == "two line":
                        k = gen_paras[1]
                        d = gen_paras[2]
                        samples = np.arange(-0.5, 0.5, 0.001).reshape((-1, 1))
                        line_points_0 = d + samples * k.reshape((1, -1))

                        d = gen_paras[3]
                        samples = np.arange(-0.5, 0.5, 0.001).reshape((-1, 1))
                        line_points_1 = d + samples * k.reshape((1, -1))

                        sample_point = np_process(inst_data[int(torch.where(primitive_ids == inst_id_1)[0])][0][0, :])
                        dis_0 = np.linalg.norm(line_points_0[0] - sample_point)
                        dis_1 = np.linalg.norm(line_points_1[0] - sample_point)

                        if dis_0 > dis_1:
                            gen_edges = np.concatenate((gen_edges, line_points_1), axis=0)
                            inter_para_set[inst_id_1][inst_id_2] = ["line", k, gen_paras[3].reshape((3,))]
                            inter_para_set[inst_id_2][inst_id_1] = ["line", k, gen_paras[3].reshape((3,))]
                        else:
                            gen_edges = np.concatenate((gen_edges, line_points_0), axis=0)
                            inter_para_set[inst_id_1][inst_id_2] = ["line", k, gen_paras[2].reshape((3,))]
                            inter_para_set[inst_id_2][inst_id_1] = ["line", k, gen_paras[2].reshape((3,))]

                    elif gen_paras[0] == "circle":
                        print(gen_paras)
                        center = gen_paras[1]
                        x_axis = gen_paras[2][0]
                        y_axis = gen_paras[2][1]
                        radius = gen_paras[3]
                        angles = np.arange(0, 2 * np.pi, 0.01).reshape((-1, 1))
                        points = radius * (np.cos(angles) * x_axis + np.sin(angles) * y_axis) + center
                        gen_edges = np.concatenate((gen_edges, points), axis=0)
                        inter_para_set[inst_id_1][inst_id_2] = ["circle", center, x_axis, y_axis, radius]
                        inter_para_set[inst_id_2][inst_id_1] = ["circle", center, x_axis, y_axis, radius]

                    elif gen_paras[0] == "ellipse":
                        ellipse_center, x_axis, y_axis, x_radius, y_radius = gen_paras[1:]
                        angles = np.arange(0, 2 * np.pi, 0.01).reshape((-1, 1))
                        points = ellipse_center + x_radius * np.cos(angles) * x_axis + y_radius * np.sin(angles) * y_axis
                        gen_edges = np.concatenate((gen_edges, points), axis=0)
                        inter_para_set[inst_id_1][inst_id_2] = ["ellipse", ellipse_center, x_axis, y_axis, x_radius,
                                                                y_radius]
                        inter_para_set[inst_id_2][inst_id_1] = ["ellipse", ellipse_center, x_axis, y_axis, x_radius,
                                                                y_radius]

                elif para1[0] == 'plane' and para2[0] == "cone":
                    gen_paras = plane_cone_inter_line(para1, para2)
                    if gen_paras[0] == "None":
                        intermap[inst_id_1, inst_id_2] = intermap[inst_id_2, inst_id_1] = False
                        continue
                    if gen_paras[0] == "circle":
                        print(gen_paras)
                        center = gen_paras[1]
                        x_axis = gen_paras[2][0]
                        y_axis = gen_paras[2][1]
                        radius = gen_paras[3]
                        angles = np.arange(0, 2 * np.pi, 0.01).reshape((-1, 1))
                        points = radius * (np.cos(angles) * x_axis + np.sin(angles) * y_axis) + center
                        gen_edges = np.concatenate((gen_edges, points), axis=0)
                        inter_para_set[inst_id_1][inst_id_2] = ["circle", center, x_axis, y_axis, radius]
                        inter_para_set[inst_id_2][inst_id_1] = ["circle", center, x_axis, y_axis, radius]

                elif para1[0] == 'cylinder' and para2[0] == "cone":
                    gen_paras = cylinder_cone_inter_line(para1, para2)
                    if gen_paras[0] == "None":
                        intermap[inst_id_1, inst_id_2] = intermap[inst_id_2, inst_id_1] = False
                        continue
                    if gen_paras[0] == "circle":
                        print(gen_paras)
                        center = gen_paras[1]
                        x_axis = gen_paras[2][0]
                        y_axis = gen_paras[2][1]
                        radius = gen_paras[3]
                        angles = np.arange(0, 2 * np.pi, 0.01).reshape((-1, 1))
                        points = radius * (np.cos(angles) * x_axis + np.sin(angles) * y_axis) + center
                        gen_edges = np.concatenate((gen_edges, points), axis=0)
                        inter_para_set[inst_id_1][inst_id_2] = ["circle", center, x_axis, y_axis, radius]
                        inter_para_set[inst_id_2][inst_id_1] = ["circle", center, x_axis, y_axis, radius]

                elif para1[0] == 'cylinder' and para2[0] == "cylinder":
                    gen_paras = cylinder_cylinder_inter_line(para1, para2)
                    if gen_paras[0] == "None":
                        intermap[inst_id_1, inst_id_2] = intermap[inst_id_2, inst_id_1] = False
                        continue

                elif para1[0] == 'plane' and para2[0] == "sphere":
                    gen_paras = plane_sphere_inter_line(para1, para2)
                    if gen_paras[0] == "None":
                        intermap[inst_id_1, inst_id_2] = intermap[inst_id_2, inst_id_1] = False
                        continue
                    if gen_paras[0] == "circle":
                        print(gen_paras)
                        center = gen_paras[1]
                        x_axis = gen_paras[2][0]
                        y_axis = gen_paras[2][1]
                        radius = gen_paras[3]
                        angles = np.arange(0, 2 * np.pi, 0.01).reshape((-1, 1))
                        points = radius * (np.cos(angles) * x_axis + np.sin(angles) * y_axis) + center
                        gen_edges = np.concatenate((gen_edges, points), axis=0)
                        inter_para_set[inst_id_1][inst_id_2] = ["circle", center, x_axis, y_axis, radius]
                        inter_para_set[inst_id_2][inst_id_1] = ["circle", center, x_axis, y_axis, radius]

                elif para1[0] == "closed-spline":
                    # todo
                    print("id {} ---> spline!!!!!".format(inst_id_1))
                    inter_para_set[inst_id_1][inst_id_2] = ["intersect-line", None, None, None]
                    inter_para_set[inst_id_2][inst_id_1] = ["intersect-line", None, None, None]
            except:
                traceback.print_exc()

    '''
    # 将分类得到的边界点聚合到最近的边上
    final_gen_edges = np.ones((0, 3), dtype=np.float64)
    raw_points = torch.from_numpy(raw_points).cuda()
    for i, item in enumerate(inst_data):
        inst_id = int(primitive_ids[i])
        print("======== cur inst {}".format(inst_id))
        edges = raw_points[(primitives == inst_id) & is_edge, :3]

        total_gen_edges = torch.ones((edges.shape[0], 0, 3), dtype=torch.float64).cuda()
        total_edge_dis = torch.ones((edges.shape[0], 0), dtype=torch.float64).cuda()

        for key in inter_para_set[inst_id]:
            if inter_para_set[inst_id][key][0] == "line":
                _gen_edges, p2l_dis = points_project_to_line(edges, inter_para_set[inst_id][key][1],
                                                             inter_para_set[inst_id][key][2])
                total_gen_edges = torch.cat((total_gen_edges, _gen_edges.reshape(-1, 1, 3)), dim=1)
                total_edge_dis = torch.cat((total_edge_dis, p2l_dis.reshape((-1, 1))), dim=1)
                print(total_gen_edges.shape, total_edge_dis.shape)

        # print(total_edge_dis.shape)
        if 0 in total_edge_dis.shape:
            continue
        small_index =  torch.argmin(total_edge_dis, dim=-1)  # N

        print(small_index.shape)

        nn_gen_edges = torch.zeros_like(edges)

        for j in range(total_gen_edges.shape[1]):
            nn_gen_edges[small_index == j, :] = total_gen_edges[small_index == j, j, :]

        final_gen_edges = np.concatenate((final_gen_edges, nn_gen_edges.cpu().numpy()), axis=0)


    np.savetxt("./dst/{}_gen_edges.txt".format(id), final_gen_edges, fmt="%0.5f", delimiter=';')
    '''

    # 求解交线之间的corner
    corner_points = np.ones((0, 3), dtype=np.float64)
    inter_edge_range = {}
    for item in primitive_ids:
        inter_edge_range[int(item)] = {}

    for i, inst in enumerate(primitive_ids):
        inst = int(inst)
        neib_insts = list(inter_para_set[inst].keys())
        print("========= cur inst {} neigbs: {}".format(inst, neib_insts))

        if len(neib_insts) < 3:
            # instance只与另外一个或两个instance相邻，则必然是circle或者ellipse
            for i in range(len(neib_insts)):
                inter_edge_range[inst][int(neib_insts[i])] = [0, float(2 * np.pi)]
                inter_edge_range[int(neib_insts[i])][inst] = [0, float(2 * np.pi)]

        else:
            for m in range(len(neib_insts) - 1):
                for n in range(m + 1, len(neib_insts)):

                    if not intermap[neib_insts[m]][neib_insts[n]] and id in ["962"]:
                        continue

                    inst_points = inst_data[int(torch.where(primitive_ids == inst)[0])][0]
                    inst_points_m = inst_data[int(torch.where(primitive_ids == neib_insts[m])[0])][0]
                    inst_points_n = inst_data[int(torch.where(primitive_ids == neib_insts[n])[0])][0]

                    if inter_para_set[inst][neib_insts[m]][0] == inter_para_set[inst][neib_insts[n]][0] == "line":
                        print("<", neib_insts[m], neib_insts[n], ">")
                        k1, k2 = inter_para_set[inst][neib_insts[m]][1], inter_para_set[inst][neib_insts[n]][1]
                        d1, d2 = inter_para_set[inst][neib_insts[m]][2], inter_para_set[inst][neib_insts[n]][2]
                        _point = line_line_inter(k1, d1, k2, d2)
                        if _point is not None:
                            if fitter_point(_point, (inst_points, inst_points_m, inst_points_n)):

                                if int(neib_insts[m]) not in inter_edge_range[inst].keys():
                                    inter_edge_range[inst][int(neib_insts[m])] = [_point]
                                    inter_edge_range[int(neib_insts[m])][inst] = [_point]
                                else:
                                    flag = True
                                    for corner in inter_edge_range[inst][int(neib_insts[m])]:
                                        if np.sqrt(np.power((corner - _point), 2).sum()) < 1e-2:
                                            flag = False
                                            break
                                    if flag:
                                        inter_edge_range[inst][int(neib_insts[m])].append(_point)
                                        inter_edge_range[int(neib_insts[m])][inst].append(_point)

                                if int(neib_insts[n]) not in inter_edge_range[inst].keys():
                                    inter_edge_range[inst][int(neib_insts[n])] = [_point]
                                    inter_edge_range[int(neib_insts[n])][inst] = [_point]
                                else:
                                    flag = True
                                    for corner in inter_edge_range[int(neib_insts[n])][inst]:
                                        if np.sqrt(np.power((corner - _point), 2).sum()) < 1e-2:
                                            flag = False
                                            break
                                    if flag:
                                        inter_edge_range[int(neib_insts[n])][inst].append(_point)
                                        inter_edge_range[inst][int(neib_insts[n])].append(_point)

                                print("{} {}: {}".format(neib_insts[m], neib_insts[n], _point))
                                corner_points = np.concatenate((corner_points, _point.reshape((1, 3))), axis=0)
                        else:
                            print("None")

                    elif inter_para_set[inst][neib_insts[m]][0] == "line" and inter_para_set[inst][neib_insts[n]][
                        0] == "circle":
                        _points = line_circle_inter(inter_para_set[inst][neib_insts[m]][1:],
                                                    inter_para_set[inst][neib_insts[n]][1:])
                        if _points is not None:
                            for _point in _points:
                                if fitter_point(_point, (inst_points, inst_points_m, inst_points_n),
                                                thresh=corner_dis_thresh):
                                    print("{} {}: {}".format(neib_insts[m], neib_insts[n], _point))
                                    corner_points = np.concatenate((corner_points, _point.reshape((1, 3))), axis=0)

                                    if int(neib_insts[m]) not in inter_edge_range[inst].keys():
                                        inter_edge_range[inst][int(neib_insts[m])] = [_point]
                                        inter_edge_range[int(neib_insts[m])][inst] = [_point]
                                    else:
                                        flag = True
                                        for corner in inter_edge_range[inst][int(neib_insts[m])]:
                                            if np.sqrt(np.power((corner - _point), 2).sum()) < 1e-2:
                                                flag = False
                                                break
                                        if flag:
                                            inter_edge_range[inst][int(neib_insts[m])].append(_point)
                                            inter_edge_range[int(neib_insts[m])][inst].append(_point)

                                    if int(neib_insts[n]) not in inter_edge_range[inst].keys():
                                        inter_edge_range[inst][int(neib_insts[n])] = [_point]
                                        inter_edge_range[int(neib_insts[n])][inst] = [_point]
                                    else:
                                        flag = True
                                        for corner in inter_edge_range[int(neib_insts[n])][inst]:
                                            if np.sqrt(np.power((corner - _point), 2).sum()) < 1e-2:
                                                flag = False
                                                break
                                        if flag:
                                            inter_edge_range[int(neib_insts[n])][inst].append(_point)
                                            inter_edge_range[inst][int(neib_insts[n])].append(_point)
                        else:
                            print("None")

                    elif inter_para_set[inst][neib_insts[m]][0] == "circle" and inter_para_set[inst][neib_insts[n]][
                        0] == "line":
                        _points = line_circle_inter(inter_para_set[inst][neib_insts[n]][1:],
                                                    inter_para_set[inst][neib_insts[m]][1:])
                        if _points is not None:
                            for _point in _points:
                                if fitter_point(_point, (inst_points, inst_points_m, inst_points_n),
                                                thresh=corner_dis_thresh):
                                    print("{} {}: {}".format(neib_insts[m], neib_insts[n], _point))
                                    corner_points = np.concatenate((corner_points, _point.reshape((1, 3))), axis=0)
                                    if int(neib_insts[m]) not in inter_edge_range[inst].keys():
                                        inter_edge_range[inst][int(neib_insts[m])] = [_point]
                                        inter_edge_range[int(neib_insts[m])][inst] = [_point]
                                    else:
                                        flag = True
                                        for corner in inter_edge_range[inst][int(neib_insts[m])]:
                                            if np.sqrt(np.power((corner - _point), 2).sum()) < 1e-2:
                                                flag = False
                                                break
                                        if flag:
                                            inter_edge_range[inst][int(neib_insts[m])].append(_point)
                                            inter_edge_range[int(neib_insts[m])][inst].append(_point)

                                    if int(neib_insts[n]) not in inter_edge_range[inst].keys():
                                        inter_edge_range[inst][int(neib_insts[n])] = [_point]
                                        inter_edge_range[int(neib_insts[n])][inst] = [_point]
                                    else:
                                        flag = True
                                        for corner in inter_edge_range[int(neib_insts[n])][inst]:
                                            if np.sqrt(np.power((corner - _point), 2).sum()) < 1e-2:
                                                flag = False
                                                break
                                        if flag:
                                            inter_edge_range[int(neib_insts[n])][inst].append(_point)
                                            inter_edge_range[inst][int(neib_insts[n])].append(_point)
                        else:
                            print("None")

    np.savetxt("./{}/{}_edges.txt".format(save_dir, id), gen_edges, fmt="%0.5f", delimiter=';')
    np.savetxt("./{}/{}_corners.txt".format(save_dir, id), corner_points, fmt="%0.5f", delimiter=';')

    final_gen_edges = np.ones((0, 3), dtype=np.float64)

    for inst_1 in range(primitive_ids.shape[0] - 1):
        for inst_2 in range(int(inst_1 + 1), int(primitive_ids.shape[0]), 1):
            print(int(primitive_ids[inst_1]), int(primitive_ids[inst_2]))
            if intermap[primitive_ids[inst_1], primitive_ids[inst_2]]:
                inst1, inst2 = int(primitive_ids[inst_1]), int(primitive_ids[inst_2])
                if inter_para_set[inst1][inst2][0] == "line":
                    d_range = inter_edge_range[inst1][inst2]
                    k, d = inter_para_set[inst1][inst2][1:3]
                    print("range", d_range)
                    range_d = []
                    for corner in d_range:
                        range_d.append(get_line_point_d(k, d, corner))
                    print("range_d", range_d)

                    if len(range_d) >= 2:
                        samples = np.arange(min(range_d), max(range_d), 0.002).reshape((-1, 1))
                        line_points = d + samples * k.reshape((1, -1))
                        final_gen_edges = np.concatenate((final_gen_edges, line_points), axis=0)

                elif inter_para_set[inst1][inst2][0] == "circle":
                    corners = inter_edge_range[inst1][inst2]
                    center, x_axis, y_axis, radius = inter_para_set[inst1][inst2][1:5]
                    if isinstance(corners[0], int):
                        # 完整圆
                        angles = np.arange(0, 2 * np.pi, 0.01).reshape((-1, 1))
                        points = radius * (np.cos(angles) * x_axis + np.sin(angles) * y_axis) + center
                        final_gen_edges = np.concatenate((final_gen_edges, points), axis=0)
                    elif isinstance(corners[0], np.ndarray):
                        inst1_points = inst_data[int(torch.where(primitive_ids == inst1)[0])][0]
                        inst2_points = inst_data[int(torch.where(primitive_ids == inst2)[0])][0]
                        # print(type(corners[0]), type(center), type(x_axis), type(y_axis), type(radius))
                        angle1, angle2 = get_circle_two_point_theta(corners[0], corners[1], center, x_axis, y_axis,
                                                                    radius, (inst1_points, inst2_points))
                        angles = np.arange(angle1, angle2, 0.01).reshape((-1, 1))
                        points = radius * (np.cos(angles) * x_axis + np.sin(angles) * y_axis) + center
                        # visualize_point_cloud(points, viz=True)
                        final_gen_edges = np.concatenate((final_gen_edges, points), axis=0)


    np.savetxt("./{}/{}_final_edges.txt".format(save_dir, id), final_gen_edges, fmt="%0.5f", delimiter=';')
