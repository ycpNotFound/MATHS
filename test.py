import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 创建示例图像数据
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# 创建图像
fig, ax = plt.subplots()

# 绘制示例图像
im = ax.imshow(data, cmap='jet')

# 使用make_axes_locatable创建坐标轴分隔器
divider = make_axes_locatable(ax)

# 指定colorbar的位置和宽度
cax = divider.append_axes("left", size="20%", pad=0.1)

# 添加colorbar，并指定ax参数
cbar = plt.colorbar(im, cax=cax)
cbar.set_ticks([])
# 显示图像
plt.show()