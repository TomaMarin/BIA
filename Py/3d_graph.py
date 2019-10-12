from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from matplotlib.widgets import Button


def custom_function(x_vals, y_vals):
    z_vals = list()
    for i in x_vals:
        z_vals.append(pow(x_vals[i], 2) + pow(y_vals[i], 2))
    return z_vals


def custom_function2(x_val, y_val):
    return pow(x_val - 3, 2) + pow(y_val - 3, 2)


def find_global_min(arrayofx):
    min_of_list = min(arrayofx, key=custom_function)
    # print(min_of_list)
    return min_of_list


def generate_normal_distribution_list(best_previous_value, scatter_value, size_of_list):
    generated_list = (np.random.normal(best_previous_value, scatter_value, size=[size_of_list, 2]))
    # print(generated_list)
    return generated_list


fitness = [0.5, 0.5]
values_difference = 0.5
amount_of_values = 10
number_of_iterations = 35

x_vals = list()
y_vals = list()
z_vals = list()

all_x_vals = list()
all_y_vals = list()
all_z_vals = list()

for i in range(number_of_iterations):
    gene_list = generate_normal_distribution_list(fitness, values_difference, amount_of_values)
    for j in range(len(gene_list)):
        all_x_vals.append(gene_list[j, 0])
        all_y_vals.append(gene_list[j, 1])
        all_z_vals.append(custom_function2(gene_list[j, 0], gene_list[j, 1]))
        if custom_function2(fitness[0], fitness[1]) > custom_function2(gene_list[j, 0], gene_list[j, 1]):
            fitness[0] = gene_list[j, 0]
            fitness[1] = gene_list[j, 1]
            x_vals.append(gene_list[j, 0])
            y_vals.append(gene_list[j, 1])
            z_vals.append(custom_function2(gene_list[j, 0], gene_list[j, 1]))
            print("true")

print(fitness)
fig = plt.figure()
ax = fig.gca(projection='3d')
# x^2 +y^2
# Make data.
X = np.arange(0, 7, 0.5)
# X = np.arange(-5, 5, 0.5)
Y = np.arange(0, 7, 0.5)
# Y = np.arange(-5, 5, 0.5)
X, Y = np.meshgrid(X, Y)
R = ((X - 3) ** 2 + (Y - 3) ** 2)
# R = (X ** 2 + Y ** 2)

# Plot the surface.

# Add a color bar which maps values to colors.

xdata = [0, 1, 2, 3, 4, 5, 6, 7]
ydata = [0, 1, 2, 3, 4, 5, 6, 7]
zdata = custom_function(xdata, ydata)

x_global = list()
x_global.append(fitness[0])
y_global = list()
y_global.append(fitness[1])
z_global = list()

z_global.append(custom_function2(fitness[0], fitness[1]))

help_x = list()
help_y = list()
help_z = list()
l, = ax.plot(x_vals, y_vals, z_vals, marker="o", markersize=9, linewidth=2)


def animate(ite):
    print(ite)
    if ite >= len(y_vals):
        help_x.clear()
        help_y.clear()
        help_z.clear()
        ani.frame_seq = ani.new_frame_seq()

    else:
        ydata = y_vals[ite]
        xdata = x_vals[ite]
        zdata = z_vals[ite]
        help_x.append(xdata)
        help_y.append(ydata)
        help_z.append(zdata)
    l.set_data_3d(help_x, help_y, help_z)
    return l,


def init():  # only required for blitting to give a clean slate.
    help_x.clear()
    help_y.clear()
    help_z.clear()
    ydata = y_vals[0]
    xdata = x_vals[0]
    zdata = z_vals[0]
    help_x.append(xdata)
    help_y.append(ydata)
    help_z.append(zdata)
    l.set_data_3d(help_x, help_y, help_z)
    return l,


ani = animation.FuncAnimation(
    fig, animate, interval=450, repeat=True)


# ax.plot(x_global, y_global, z_global, marker="*", markersize=12, linewidth=3)
surf = ax.plot_surface(X, Y, R, color='r',
                       linewidth=0, antialiased=False)
plt.show()
