# import matplotlib.pyplot as plt
# import numpy as np
#
#
# methods = ['FPS', 'Random', 'Poisson', 'Denoising', 'Ours']
# x = np.linspace(0, 6, 5)
#
#
# # fps_y = np.sin(x)
# # random_y = np.cos(x)
# # poisson_y = x / 3 + np.random.normal(size=x.shape[0], scale=0.1)
# # denoising_y = x**2 / 40 + np.random.normal(size=x.shape[0], scale=0.1)
# # ours_y = (x**3)/80 + np.random.normal(size=x.shape[0], scale=0.1)
#
# fps_y = np.asarray([3.9, 4.5, 6.1, 6.95, 8.2])
# random_y = np.asarray([3.85, 4.33, 4.85, 5.95, 6.8])
# poisson_y = np.asarray([3.77, 4.22, 4.55, 4.67, 5.12])
# denoising_y = np.asarray([3.70, 4.1, 4.17, 4.25, 4.55])
# ours_y = np.asarray([3.66, 3.91, 4.06, 4.12, 4.3])
#
# # 设置字体大小
# plt.rcParams.update({'font.size': 14})
#
# fig, ax = plt.subplots()
#
# # 绘制每种方法的线条
# for method, y in zip(methods, [fps_y, random_y, poisson_y, denoising_y, ours_y]):
#     ax.plot(x, y, label=method)
#
# # 添加标题和标签
# # ax.set_title('')
# ax.set_xlabel('noise level (%)')
# ax.set_ylabel('chamfer distance (e-3)')
#
# # 添加图例
# legend = ax.legend(loc='upper left', shadow=True, fontsize='medium')
#
# # 设置网格线
# ax.grid(True)
#
# # 显示图表
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
# 假设这是你的数据
x_values = np.linspace(0, 0.02, 9) # [0, 0.0005, 0.001, 0.0015, 0.002]
y1_values = [4.76, 4.5, 4.32, 3.95, 3.87, 3.875, 3.89, 3.92, 4.0] # [3.89, 3.88, 3.87, 3.88, 4.56] # [3.95, 3.915, 3.87, 3.895, 3.933]  # 倒角距离
y2_values = [] # [0.777, 0.78, 0.783, 0.781, 0.76] # [0.785, 0.7844, 0.7846, 0.7832, 0.7822]  # 网格质量

# 创建figure对象和子图
fig, ax1 = plt.subplots()

# 在第一个y轴上绘制数据系列
color = 'tab:red'
ax1.set_xlabel('')  # X轴标签
ax1.set_ylabel('chamfer distance (e-3)', fontsize=16)  # 第一个y轴的标签
ax = fig.gca()
ax.set_xlim([3.8, 4.0])
ax1.plot(x_values, y1_values)
ax1.tick_params(axis='y', which='major', labelcolor=color)

# 创建第二个y轴
# ax2 = ax1.twinx()
# color = 'tab:blue'  # 第二个y轴的颜色
# ax2.set_ylabel('G', color=color, fontsize=16)  # 第二个y轴的标签
# ax2.plot(x_values, y2_values, color=color)
# ax2.tick_params(axis='y', which='major', labelcolor=color)
# plt.rcParams.update({'font.size': 14})
# 调整图表的其他属性
# plt.title('Title of the Chart')  # 图表标题
plt.xlabel('')  # X轴标签
plt.xticks([0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02])  # 自定义X轴刻度
plt.xlim(0, 0.02)  # 设置X轴范围
plt.grid(True)  # 开启网格线

# 显示图表
plt.show()