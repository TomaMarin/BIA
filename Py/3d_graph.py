from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from matplotlib.widgets import Button


def custom_function(x_vals, y_vals):
    z_vals = list()
    for i in x_vals:
        z_vals.append(pow(x_vals[i], 2) + pow(y_vals[i], 2))
    return z_vals


def custom_function2(x_vals, y_vals):
    z_vals = list()
    for i in x_vals:
        z_vals.append(pow(x_vals[i]-3, 2) + pow(y_vals[i]-3, 2))
    return z_vals


fig = plt.figure()
ax = fig.gca(projection='3d')
# x^2 +y^2
# Make data.
# X = np.arange(0, 10, 0.5)
X = np.arange(-5, 5, 0.5)
# Y = np.arange(0, 10, 0.5)
Y = np.arange(-5, 5, 0.5)
X, Y = np.meshgrid(X, Y)
# R = ((X-3)**2 + (Y-3)**2)
R = (X ** 2 + Y ** 2)

# Plot the surface.
surf = ax.plot_surface(X, Y, R, color='r',
                       linewidth=0, antialiased=False)
# br = ax.plot([3], [3], 'r*', markersize=15)
# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.

xdata = [0, 1, 2, 3, 4, 5, 6, 7]
ydata = [0, 1, 2, 3, 4, 5, 6, 7]
zdata = custom_function(xdata, ydata)


fig.colorbar(surf, shrink=0.5, aspect=2)

ax.plot(xdata, ydata, zdata, marker="o")
plt.show()
