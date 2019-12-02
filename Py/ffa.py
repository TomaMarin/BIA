import numpy as np
import matplotlib.pyplot as plt
from operator import attrgetter
from mpl_toolkits.mplot3d import Axes3D
import copy
import random
from matplotlib import cm, animation
import math
from statistics import mean


def schaffer_n2(x):
    x1 = x[0]
    x2 = x[1]
    poweer = (pow(x1, 2) - pow(x2, 2))
    poweer2 = (pow(x1, 2) + pow(x2, 2))
    result = 0.5 + ((pow(np.sin(poweer), 2) - 0.5)
                    / (pow((1 + 0.001 * (poweer2)), 2))
                    )
    return result


def calculate_distance_between_points(actual_point, point):
    result = 0
    for i in range(len(actual_point)):
        result += pow(actual_point[i] - point[i], 2)
    return np.sqrt(result)


def custom_function(x):
    x1 = x[0]
    x2 = x[1]
    return pow(x1 - 3, 2) + pow(x2 - 3, 2)


def schwefel_function(x):
    total_f_val = 0.0
    for i in x:
        total_f_val += i * np.sin(np.sqrt(np.abs(i)))
    return (418.9829 * 2) - total_f_val


class FireFly:
    function_value = 0

    def __init__(self, parameters, function_value):
        self.parameters = parameters
        self.function_value = function_value

    def __repr__(self):
        return "I am at " + str(self.parameters) + " with f val " + str(self.function_value)


def create_init_generation(D, pop_size):
    init_pop = list()
    for i in range(pop_size):
        actual_params = (random.sample(range(-100, 100), D))
        init_pop.append(
            FireFly(actual_params, schaffer_n2(actual_params)))
    return init_pop


def calc_attractiveness(distance, beta):
    psi = 1.0
    return beta * (1.0 / (psi + distance))


def movement_of_ff(brighter_ff: FireFly, dimmer_ff: FireFly, alpha, beta):
    r = calculate_distance_between_points(brighter_ff.parameters, dimmer_ff.parameters)
    new_ff_w_params = copy.deepcopy(dimmer_ff)
    for i in range(len(new_ff_w_params.parameters)):
        new_ff_w_params.parameters[i] = new_ff_w_params.parameters[i] + calc_attractiveness(r, beta) * (
                brighter_ff.parameters[i] - new_ff_w_params.parameters[i]) + alpha * (random.random() - 0.5)
        if new_ff_w_params.parameters[i] > 100:
            new_ff_w_params.parameters[i] = 100 - random.randint(5, 15)
        elif new_ff_w_params.parameters[i] < -100:
            new_ff_w_params.parameters[i] = -100 + random.randint(5, 15)
    new_ff_w_params.function_value = schaffer_n2(new_ff_w_params.parameters)
    return new_ff_w_params


def movement_of_best_ff(best_ff: FireFly, alpha):
    new_params_w_ff = copy.deepcopy(best_ff)
    for i in range(len(new_params_w_ff.parameters)):
        new_params_w_ff.parameters[i] = new_params_w_ff.parameters[i] + alpha * (random.random() - 0.5)
        if new_params_w_ff.parameters[i] > 100:
            new_params_w_ff.parameters[i] = 100 - random.randint(5, 15)
        elif new_params_w_ff.parameters[i] < -100:
            new_params_w_ff.parameters[i] = -100 + random.randint(5, 15)
        new_params_w_ff.function_value = schaffer_n2(new_params_w_ff.parameters)
    return new_params_w_ff


all_iterations = list()

all_x_vals = list()
all_y_vals = list()
all_z_vals = list()


def ffa(D, number_of_iterations, pop_size, alpha, beta):
    init_gen = create_init_generation(D, pop_size)
    current_pop = init_gen[:]
    current_pop = sorted(current_pop, key=attrgetter('function_value'))
    for i in current_pop:
        all_x_vals.append(i.parameters[0])
        all_y_vals.append(i.parameters[1])
        all_z_vals.append(i.function_value)
    for i in range(number_of_iterations):
        next_gen = list()
        for j in range(1, len(current_pop)):
            for k in range(1, len(current_pop)):
                if current_pop[j].function_value < current_pop[k].function_value:
                    current_pop[k] = movement_of_ff(current_pop[j], current_pop[k], alpha, beta)
        current_pop[0] = movement_of_best_ff(current_pop[0], alpha)
        print("Ite: ", i, " best ff is : ", current_pop[0])
        next_gen.extend(current_pop[:])
        current_pop.clear()
        current_pop = copy.deepcopy(next_gen[:])
        current_pop = sorted(current_pop[:], key=attrgetter('function_value'))
        all_iterations.append(current_pop[:])


help_x = list()
help_y = list()
help_z = list()
fig = plt.figure()
ax = fig.gca(projection='3d')
v, = ax.plot(all_x_vals, all_y_vals, all_z_vals, linestyle="", color='r', markersize=12, marker="o", linewidth=12)

def animate_vals(ite):
    # print(ite, " vals")
    plt.title("Migrace  number: " + str(ite))
    if ite >= len(all_iterations):
        help_x.clear()
        help_y.clear()
        help_z.clear()
        ani_vals.frame_seq = ani_vals.new_frame_seq()
    else:
        help_x.clear()
        help_y.clear()
        help_z.clear()
        for i in all_iterations[ite]:
            help_x.append(i.parameters[0])
            help_y.append(i.parameters[1])
            help_z.append(i.function_value)
        # ydata = y_leaders_vals[ite]
        # xdata = x_leaders_vals[ite]
        # zdata = z_leaders_vals[ite]
        # help_x.append(xdata)
        # help_y.append(ydata)
        # help_z.append(zdata)
    v.set_data_3d(help_x, help_y, help_z)
    return v,


def init_vals():  # only required for blitting to give a clean slate.
    help_x.clear()
    help_y.clear()
    help_z.clear()
    for i in all_y_vals:
        help_x.append(i)
    for i in all_y_vals:
        help_y.append(i)
    for i in all_y_vals:
        help_z.append(i)
    v.set_data_3d(help_x, help_y, help_z)
    return v,


dimension = 2
number_of_iterations = 50
pop_size = 15
alpha = 0.3
beta = 1.9

ffa(dimension, number_of_iterations, pop_size, alpha, beta)

X = np.arange(-100, 100, 1)
Y = np.arange(-100, 100, 1)
X, Y = np.meshgrid(X, Y)
# R = ((X - 3) ** 2 + (Y - 3) ** 2)
ani_vals = animation.FuncAnimation(fig, animate_vals, interval=300, init_func=init_vals,  repeat=True)
arr = [X, Y]
R = (schaffer_n2(arr))
surf = ax.plot_surface(X, Y, R, color='b',
                       linewidth=0, antialiased=True, alpha=0.1)

plt.show()
