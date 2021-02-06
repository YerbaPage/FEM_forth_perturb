from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt
from utils import *
fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(-1, 1, 0.01)
y = np.arange(-1, 1, 0.01)
X, Y = np.meshgrid(x, y)  # 网格的创建，生成二维数组，这个是关键
Z = exact_u(X, Y)
plt.xlabel('x')
plt.ylabel('y')

ax.plot_surface(X, Y, Z, rstride=2, cstride=1, cmap='rainbow')
plt.show()