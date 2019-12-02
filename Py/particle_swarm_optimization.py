import numpy as np
import matplotlib.pyplot as plt
from operator import attrgetter
from mpl_toolkits.mplot3d import Axes3D
import copy
import random
from matplotlib import cm, animation
import math


def schwefel_function(x):
    total_f_val = 0.0
    for i in x:
        # f7(x)=sum(-x(i)·sin(sqrt(abs(x(i))))), i=1:n; -500<=x(i)<=500.
        total_f_val += -i * np.sin(np.sqrt(np.abs(i)))
    return total_f_val


class Particle:
    velocity = np.array([0, 0])
    pBest = 0
    function_value = 0

    def __init__(self, parameters, pBest, velocity, function_value):
        self.parameters = parameters
        self.pBest = parameters
        self.velocity = velocity
        self.function_value = function_value

    def __repr__(self):
        return "I am at " + str(self.parameters) + " my pbest is " + str(self.pBest) + " with velocity " + str(
            self.velocity) + " with f val " + str(self.function_value)

    def set_pbest(self, old_params):
        if schwefel_function(self.parameters) > schwefel_function(old_params):
            self.pBest = old_params[:]


def init_generation(D, pop_size, max_velocity):
    init_pop = list()
    for i in range(pop_size):
        actual_params = (random.sample(range(-500, 500), D))
        velocity = [random.uniform(0.0, 0.5) for iter in range(D)]
        init_pop.append(
            Particle(actual_params, 0, velocity, schwefel_function(actual_params)))
    return init_pop


def calculate_velocity(particle: Particle, gBest: Particle, c1, c2, w):
    new_velocity = particle.velocity[:]
    for i in range(len(particle.velocity)):
        new_velocity[i] = w * particle.velocity[i] + c1 * np.random.random(1)[0] * (
                abs(particle.pBest[i]) - abs(particle.parameters[i])) + c2 * \
                          np.random.random(1)[0] * (abs(gBest.parameters[i]) - abs(particle.parameters[i]))
        if new_velocity[i] < -100:
            new_velocity[i] = -100
        if new_velocity[i] > 100:
            new_velocity[i] = 100
    return new_velocity


def move(parameters, velocity):
    old_params = parameters[:]
    for i in range(len(parameters)):
        parameters[i] = parameters[i] + velocity[i]
        if parameters[i] > 500:
            parameters[i] = 500 - random.randint(10, 49)
        if parameters[i] < -500:
            parameters[i] = 500 + random.randint(10, 49)
    return parameters


def set_p_best(pbest_params, new_params):
    if schwefel_function(pbest_params) > schwefel_function(new_params):
        return new_params
    return pbest_params


gBest_per_every_migration = list()
x_gBests_vals = list()
y_gBests_vals = list()
z_gBests_vals = list()
x_vals = list()
y_vals = list()
z_vals = list()
all_x_vals = list()
all_y_vals = list()
all_z_vals = list()
best_individuals_per_every_migration = list(list())


def pso(dimension, max_velocity, pop_size, iterations, c1, c2):
    current_pop = init_generation(dimension, pop_size, max_velocity)
    best_individuals_per_every_migration.append(current_pop[:])
    for i in current_pop:
        all_x_vals.append(i.parameters[0])
        all_y_vals.append(i.parameters[1])
        all_z_vals.append(i.function_value)
    for i in range(iterations):
        migrace_pop = list()
        gBest = min(current_pop, key=attrgetter('function_value'))
        print(gBest)
        gBest_per_every_migration.append(gBest)
        x_gBests_vals.append(gBest.parameters[0])
        y_gBests_vals.append(gBest.parameters[1])
        z_gBests_vals.append(gBest.function_value)
        for j in range(len(current_pop)):
            # if current_pop[j] != gBest:
                new_velocity = calculate_velocity(current_pop[j], gBest, c1, c2, 0.5)
                new_params = move(current_pop[j].parameters, new_velocity)
                new_pbest = set_p_best(current_pop[j].pBest, new_params)
                new_particle = Particle(new_params, new_pbest, new_velocity, schwefel_function(new_params))
                migrace_pop.append(new_particle)
        best_individuals_per_every_migration.append(current_pop[:])
        # migrace_pop.append(gBest)
        current_pop = copy.deepcopy(migrace_pop)
        for j in range(pop_size):
            x_vals.append(current_pop[j].parameters[0])
            y_vals.append(current_pop[j].parameters[1])
            z_vals.append(current_pop[j].function_value)


fig = plt.figure()
ax = fig.gca(projection='3d')
help_x = list()
help_y = list()
help_z = list()
help_x_gbest = list()
help_y_gbest = list()
help_z_gbest = list()
l, = ax.plot(x_gBests_vals, y_gBests_vals, z_gBests_vals, linestyle="", color='y', markersize=20, marker="X",
             linewidth=20)
v, = ax.plot(all_x_vals, all_y_vals, all_z_vals, linestyle="", color='r', markersize=12, marker="o", linewidth=12)


# def animate(ite):
#     print(ite, " leader")
#     plt.title("Migrace  number: " + str(ite))
#     if ite >= len(gBest_per_every_migration):
#         help_x_gbest.clear()
#         help_y_gbest.clear()
#         help_z_gbest.clear()
#         ani.frame_seq = ani.new_frame_seq()
#
#     else:
#         help_x_gbest.clear()
#         help_y_gbest.clear()
#         help_z_gbest.clear()
#         ydata = y_gBests_vals[ite]
#         xdata = x_gBests_vals[ite]
#         zdata = z_gBests_vals[ite]
#         help_x_gbest.append(xdata)
#         help_y_gbest.append(ydata)
#         help_z_gbest.append(zdata)
#     l.set_data_3d(help_x_gbest, help_y_gbest, help_z_gbest)
#     return l,


def animate_vals(ite):
    # print(ite, " vals")
    plt.title("Migrace  number: " + str(ite))
    if ite >= len(best_individuals_per_every_migration):
        help_x.clear()
        help_y.clear()
        help_z.clear()
        ani_vals.frame_seq = ani_vals.new_frame_seq()
    else:
        help_x.clear()
        help_y.clear()
        help_z.clear()
        for i in best_individuals_per_every_migration[ite]:
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


def init_leader():  # only required for blitting to give a clean slate.
    help_x_gbest.clear()
    help_y_gbest.clear()
    help_z_gbest.clear()
    ydata = y_gBests_vals[0]
    xdata = x_gBests_vals[0]
    zdata = z_gBests_vals[0]
    help_x_gbest.append(xdata)
    help_y_gbest.append(ydata)
    help_z_gbest.append(zdata)
    l.set_data_3d(help_x_gbest, help_y_gbest, help_z_gbest)
    return l,


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


pso(2, 2, 20, 20, 2, 2)

X = np.arange(-500, 500, 5)
Y = np.arange(-500, 500, 5)
X, Y = np.meshgrid(X, Y)
# R = ((X - 3) ** 2 + (Y - 3) ** 2)
arr = [X, Y]
R = (schwefel_function(arr))
ani_vals = animation.FuncAnimation(fig, animate_vals, interval=700, init_func=init_vals,  repeat=True)
# ani = animation.FuncAnimation(
#     fig, animate, init_func=init_leader, interval=775, repeat=True)
surf = ax.plot_surface(X, Y, R, color='b',
                       linewidth=0, antialiased=True, alpha=0.1)
plt.show()
