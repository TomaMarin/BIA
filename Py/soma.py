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


0


def custom_function(x):
    x1 = x[0]
    x2 = x[1]
    return pow(x1 - 3, 2) + pow(x2 - 3, 2)


class Individual:
    function_value = 0.0

    def __init__(self, parameters, function_value):
        self.parameters = parameters
        self.function_value = function_value

    def __repr__(self):
        return "Individual with parametres on " + str(self.parameters) + " and f val " + str(self.function_value)


x_vals = list()
y_vals = list()
z_vals = list()

x_leaders_vals = list()
y_leaders_vals = list()
z_leaders_vals = list()

best_individuals_per_every_migration = list(list())
leaders_per_every_migration = list()


def SOMA(D, number_of_migrations, pop_size, PRT, path_length):
    current_pop = init(D, pop_size)
    migration_number = 0
    best_individuals_per_every_migration.append(current_pop[:])
    centroids = setKMeansForFirstTime(3, current_pop)
    for fl in centroids:
        x_leaders_vals.append(fl.parameters[0])
        y_leaders_vals.append(fl.parameters[1])
        z_leaders_vals.append(fl.function_value)
    for i in current_pop:
        all_x_vals.append(i.parameters[0])
        all_y_vals.append(i.parameters[1])
        all_z_vals.append(i.function_value)
    while migration_number < number_of_migrations:
        # leader = min(current_pop, key=attrgetter('function_value'))
        if migration_number > 0:
            centroids = setCentroidsForKmeans(centroids, current_pop)
        leaders_per_every_migration.append(centroids)
        # for leader in centroids:
        #     x_leaders_vals.append(leader.parameters[0])
        #     y_leaders_vals.append(leader.parameters[1])
        #     z_leaders_vals.append(leader.function_value)
        print("leader f vals : ", centroids)
        traveling_pop = list()
        traveling_pop.clear()
        for i in range(pop_size):
            if not centroids.__contains__(current_pop[i]):
                individual_parameters = travel(D, PRT, path_length, current_pop[i],
                                               find_centroid_for_item(centroids, current_pop[i]))
                traveling_pop.append(Individual(individual_parameters, schwefel_function(individual_parameters)))
                # traveling_pop.append(Individual(individual_parameters, custom_function(individual_parameters)))
        current_pop.clear()
        current_pop = copy.deepcopy(traveling_pop)
        best_individuals_per_every_migration.append(current_pop[:])
        # current_pop.(ce)
        for j in centroids:
            current_pop.append(j)
        migration_number += 1
    for k in range(pop_size):
        x_vals.append(current_pop[k].parameters[0])
        y_vals.append(current_pop[k].parameters[1])
        z_vals.append(current_pop[k].function_value)


def init(D, pop_size):
    current_pop = list()

    for i in range(pop_size):
        individual_params = random.sample(range(-500, 500), D)
        # individual_params = np.random.normal(0, 500, size=[1, D])
        new_individual = Individual(individual_params, schwefel_function(individual_params))
        # new_individual = Individual(individual_params[0], schwefel_function(individual_params[0]))
        # new_individual = Individual(individual_params[0], custom_function(individual_params[0]))
        current_pop.append(new_individual)
    return current_pop


def set_PTV_of_individual(D, PRT):
    ptv = np.empty(D, dtype=object)
    for i in range(D):
        if np.random.random() < PRT:
            ptv[i] = 1
        else:
            ptv[i] = 0
    return ptv


def travel(D, PRT, path_length, individual: Individual, leader: Individual):
    ptv = set_PTV_of_individual(D, PRT)
    leader_positions = copy.deepcopy(leader.parameters)
    actual_positions = copy.deepcopy(individual.parameters)
    start_positions = copy.deepcopy(individual.parameters)
    best_positions = copy.deepcopy(individual.parameters)
    t = 0.11
    step = 0.11
    while t < path_length:
        for i in range(D):
            actual_positions[i] = start_positions[i] + (leader_positions[i] - start_positions[i]) * t * ptv[i]
            if actual_positions[i] > 500 or actual_positions[i] < -500:
                actual_positions[i] = actual_positions[i - 1]
                break
        if schwefel_function(actual_positions) <= schwefel_function(best_positions):
            # if custom_function(actual_positions) <= custom_function(best_positions):
            # print(custom_function(actual_positions), "actual")
            # print(custom_function(best_positions), "best")
            # best_positions = actual_positions[:]
            best_positions = copy.deepcopy(actual_positions)
        t += step
    return best_positions


def setKMeansForFirstTime(number_of_centroids, vals: list):
    centroids = list()
    for i in range(number_of_centroids):
        centroids.append(vals[int(random.random() * len(vals) - 1)])
    return centroids


def setCentroidsForKmeans(centroids: list, vals: list):
    new_centroids = copy.deepcopy(centroids)
    for i in range(len(centroids)):
        number_of_items = 0
        for j in vals:
            if find_centroid_for_item(centroids, j) == centroids[i]:
                number_of_items += 1
                new_centroids[i].parameters[0] = new_centroids[i].parameters[0] + j.parameters[0]
                new_centroids[i].parameters[1] = new_centroids[i].parameters[1] + j.parameters[1]
        for n in range(len(new_centroids[i].parameters)):
            if new_centroids[i].parameters[n] / number_of_items > 500:
                new_centroids[i].parameters[n] = int(random.random() * 500 - 1)
            if new_centroids[i].parameters[n] / number_of_items < -500:
                new_centroids[i].parameters[n] = int(random.random() * -500 - 1)
            else:
                new_centroids[i].parameters[n] = new_centroids[i].parameters[n] / number_of_items

    return new_centroids


# private static Double calculateEuclidanDistanceFor4D(Double x1, Double x2, Double x3, Double x4, Double y1, Double y2, Double y3, Double y4) {
#         return Math.sqrt((Math.pow(x1 - y1, 2)) + (Math.pow(x2 - y2, 2)) + (Math.pow(x3 - y3, 2)) + (Math.pow(x4 - y4, 2)));
#     }

def calcEuclidanDistance(x1, x2, y1, y2, ):
    return np.sqrt((np.power(x1 - y1, 2)) + (np.power(x2 - y2, 2)))


def find_centroid_for_item(centroids: list, item: Individual):
    best_centroid_distance = 1000.0
    best_centroid = centroids[0]
    for i in centroids:
        if (calcEuclidanDistance(item.parameters[0], item.parameters[1], i.parameters[0],
                                 i.parameters[1]) < best_centroid_distance):
            best_centroid_distance = calcEuclidanDistance(item.parameters[0], item.parameters[1],
                                                          i.parameters[0], i.parameters[1])
            best_centroid = i

    return best_centroid


help_x = list()
help_y = list()
help_z = list()

help_x_leader = list()
help_y_leader = list()
help_z_leader = list()
fig = plt.figure()
ax = fig.gca(projection='3d')

all_x_vals = list()
all_y_vals = list()
# all_z_vals = np.zeros(len(best_individuals_per_every_migration))
all_z_vals = list()
l, = ax.plot(x_leaders_vals, y_leaders_vals, z_leaders_vals, linestyle="", color='y', markersize=20, marker="X",
             linewidth=20)
# for i in range(len(best_individuals_per_every_migration)):
# all_x_vals[i] = best_individuals_per_every_migration[]
v, = ax.plot(all_x_vals, all_y_vals, all_z_vals, linestyle="", color='r', markersize=12, marker="o", linewidth=12)


def animate(ite):
    # print(ite," leader")
    plt.title("Migrace  number: " + str(ite))
    if ite >= len(leaders_per_every_migration):
        help_x_leader.clear()
        help_y_leader.clear()
        help_z_leader.clear()
        ani.frame_seq = ani.new_frame_seq()

    else:
        help_x_leader.clear()
        help_y_leader.clear()
        help_z_leader.clear()
        for i in leaders_per_every_migration[ite]:
            help_x_leader.append(i.parameters[0])
            help_y_leader.append(i.parameters[1])
            help_z_leader.append(i.function_value)
    l.set_data_3d(help_x_leader, help_y_leader, help_z_leader)
    return l,


def animate_vals(ite):
    # print(ite," vals")
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
    v.set_data_3d(help_x, help_y, help_z)
    return v,


# def init_g():  # only required for blitting to give a clean slate.
#     help_x.clear()
#     help_y.clear()
#     help_z.clear()
#     ydata = y_vals[0]
#     xdata = x_vals[0]
#     zdata = z_vals[0]
#     help_x.append(xdata)
#     help_y.append(ydata)
#     help_z.append(zdata)
#     l.set_data_3d(help_x, help_y, help_z)
#     return l,


(SOMA(2, 60, 30, 0.1, 2))

# ax.scatter(x_vals, y_vals, z_vals, color='g', marker="x", linewidth=7)
X = np.arange(-500, 500, 5)
Y = np.arange(-500, 500, 5)
X, Y = np.meshgrid(X, Y)
# R = ((X - 3) ** 2 + (Y - 3) ** 2)
arr = [X, Y]
R = (schwefel_function(arr))
ani_vals = animation.FuncAnimation(fig, animate_vals, interval=550, repeat=False)
ani = animation.FuncAnimation(
    fig, animate, interval=550, repeat=False)
surf = ax.plot_surface(X, Y, R, color='b',
                       linewidth=0, antialiased=True, alpha=0.1)
plt.show()
