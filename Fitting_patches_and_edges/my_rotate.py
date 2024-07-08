import numpy as np

from utils import draw_geometries, visualize_point_cloud
import open3d

if __name__ == "__main__":
    p1 = np.loadtxt("D:/modelnet40_seg/chair_0890_points.xyz", delimiter=" ")
    p2 = np.loadtxt("D:/shapenetcorev1/Chair_pred/1ab8a3b55c14a7b27eaeab1f0c9120b7_points.xyz", delimiter=" ")

    rotation_angle = -np.pi / 2
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    R = np.array([
        [cosval, sinval, 0],
        [-sinval, cosval, 0],
        [0, 0, 1]
    ])
    p1 = np.dot(p1[:, :3], R)
    p1[:, 0] = -p1[:, 0]
    visualize_point_cloud(p1, viz=True)
    visualize_point_cloud(p2, viz=True)
    '''
    p1 = open3d.io.read_point_cloud("D:/modelnet40_seg/chair_0890_points.xyz")
    p2 = open3d.io.read_point_cloud("D:/shapenetcorev1/Chair_pred/1ab8a3b55c14a7b27eaeab1f0c9120b7_points.xyz")


    R = p1.get_rotation_matrix_from_xyz((0, 0, -np.pi / 2))
    p1.rotate(R, (0, 0, 0))
    

    open3d.visualization.draw_geometries([p1],
                                      window_name="",
                                      point_show_normal=False,
                                      width=800,
                                      height=600)

    open3d.visualization.draw_geometries([p2],
                                      window_name="",
                                      point_show_normal=False,
                                      width=800,
                                      height=600)'''