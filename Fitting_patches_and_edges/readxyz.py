import numpy as np
import random


def load_point_cloud_from_xyz(file_path):
    """从.xyz文件加载点云数据"""
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if True: # len(parts) == 3:  # 假设每一行有3个浮点数，分别代表x, y, z
                point = [float(coord) for coord in parts]
                points.append(point[:3])
    return np.array(points)


def add_noise_to_point_cloud(point_cloud, noise_std=0.01):
    """给点云中的每个点添加随机噪声"""
    noise = np.random.normal(0, noise_std, point_cloud.shape)
    noisy_point_cloud = point_cloud + noise
    return noisy_point_cloud


def save_point_cloud_to_xyz(point_cloud, file_path):
    """保存点云到.xyz文件"""
    with open(file_path, 'w') as file:
        for point in point_cloud:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")


def main():
    input_file_path = 'C:\\Users\\Sean\\Desktop\\毕设工作\\论文大改\\降采样\\mops\\my.xyz'   # 1024-128-newmops-13.xyz'  # 输入文件路径，请根据实际情况修改
    output_file_path = 'C:\\Users\\Sean\\Desktop\\毕设工作\\论文大改\\降采样\\mops\\my_new.xyz' # 1024-128-newmops-13_new.xyz'  # 输出文件路径
    noise_standard_deviation = 0.0035  # 噪声的标准差

    # 加载点云
    point_cloud = load_point_cloud_from_xyz(input_file_path)
    print(f"Loaded point cloud with {len(point_cloud)} points.")

    # 添加噪声
    noisy_point_cloud = add_noise_to_point_cloud(point_cloud, noise_standard_deviation)

    # 保存带有噪声的点云
    save_point_cloud_to_xyz(noisy_point_cloud, output_file_path)
    print(f"Noisy point cloud saved to {output_file_path}.")


if __name__ == "__main__":
    main()
