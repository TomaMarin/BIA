import math
import matplotlib.pyplot as plt
from matplotlib import cm, animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def custom_function(x):
    x1 = x[0]
    x2 = x[1]
    return pow(x1 - 3, 2) + pow(x2 - 3, 2)


def testing_function(x1, x2):
    obj = 0.2 + x1 ** 2 + x2 ** 2 - 0.1 * math.cos(6.0 * 3.1415 * x1) - 0.1 * math.cos(6.0 * 3.1415 * x2)
    return obj


def testing_function2(x):
    obj = pow(x, 3) - 6 * pow(x, 2) - 15 * x + 100
    return obj


def testing_function3(x):
    obj = pow(x, 4) - 2 * pow(x, 2) - 1
    return obj


def generate_normal_distribution_obj(best_previous_value, scatter_value, size_of_list, number_of_dimension):
    generated_obj = (np.random.normal(best_previous_value, scatter_value, size=[size_of_list, number_of_dimension]))
    # print(generated_list)
    return generated_obj


def acceptance_probability(cost_value, new_cost_value, actual_t):
    if new_cost_value < cost_value:
        if new_cost_value > 0:
            return 1
    else:
        p = np.exp(- np.abs((new_cost_value - cost_value)) / actual_t)
        return p


maxT = 1

decreaseT = 0.01

actualT = maxT

minT = 0.02
#
costValue = [0.5, 0.5]
# costValue = -3
# costValue = 0

values_difference = 0.5
amount_of_values = 1
# number_of_iterations = 35

dimension = 2

x_vals = list()
y_vals = list()
z_vals = list()

# while actualT > minT:
#     geneObj = generate_normal_distribution_obj(costValue, values_difference, amount_of_values)
#     if custom_function(geneObj[0, 0], geneObj[0, 1]) < custom_function(costValue[0], costValue[1]):
#         costValue[0] = geneObj[0, 0]
#         costValue[1] = geneObj[0, 1]
#     actualT = actualT - actualT * decreaseT
#

# fnc (x-3)^2
while actualT > minT:
    geneObj = generate_normal_distribution_obj(costValue, values_difference, amount_of_values, dimension)

    if acceptance_probability(custom_function(costValue),
                              custom_function(geneObj[0]), actualT) > np.random.random():
        x_vals.append(geneObj[0, 0])
        y_vals.append(geneObj[0, 1])
        z_vals.append(custom_function(geneObj[0]))
        costValue = geneObj[0]

    actualT = actualT - (actualT * decreaseT)
print(costValue, "with function val:", custom_function(costValue))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x_vals, y_vals, z_vals, marker="o", markersize=9, linewidth=2)

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
zdata = custom_function(xdata)


help_x = list()
help_y = list()
help_z = list()
l, = ax.plot(x_vals, y_vals, z_vals, marker="o", markersize=9, linewidth=2)


def animate(ite):
    # print(ite)
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

surf = ax.plot_surface(X, Y, R, color='r',
                       linewidth=0, antialiased=True, alpha=0.5)
# fnc testing f 3
# while actualT > minT:
#     geneObj = generate_normal_distribution_obj(costValue, values_difference, amount_of_values, dimension)
#
#     if acceptance_probability(testing_function3(costValue),
#                               testing_function3(geneObj[0]), actualT) > np.random.random():
#         costValue = geneObj[0]
#
#     actualT = actualT - (actualT * decreaseT)

# print(costValue, "with function val:", testing_function3(costValue))


plt.show()
