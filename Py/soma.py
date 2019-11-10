import numpy as np
import matplotlib.pyplot as plt
from operator import attrgetter
from mpl_toolkits.mplot3d import Axes3D
import copy
import random


def schwefel_function(x):
    total_f_val = 0.0
    for i in x:
        #         f7(x)=sum(-x(i)·sin(sqrt(abs(x(i))))), i=1:n; -500<=x(i)<=500.
        total_f_val += -i * np.sin(np.sqrt(np.abs(i)))
    return total_f_val


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
all_x_vals = list()
all_y_vals = list()
all_z_vals = list()
x_leaders_vals = list()
y_leaders_vals = list()
z_leaders_vals = list()

best_individuals_per_every_migration = list(list())
leader_per_every_migration = list()


def SOMA(D, number_of_migrations, pop_size, PRT, path_length):
    current_pop = init(D, pop_size)
    migration_number = 0
    best_individuals_per_every_migration.append(current_pop[:])
    while migration_number < number_of_migrations:
        leader = min(current_pop, key=attrgetter('function_value'))
        leader_per_every_migration.append(leader)
        x_leaders_vals.append(leader.parameters[0])
        y_leaders_vals.append(leader.parameters[1])
        z_leaders_vals.append(leader.function_value)
        print("leader f val : ", leader.function_value)
        traveling_pop = list()
        traveling_pop.clear()
        for i in range(pop_size):
            if current_pop[i] != leader:
                individual_parameters = travel(D, PRT, path_length, current_pop[i], leader)
                traveling_pop.append(Individual(individual_parameters, schwefel_function(individual_parameters)))
                # traveling_pop.append(Individual(individual_parameters, custom_function(individual_parameters)))
        current_pop.clear()
        current_pop = copy.deepcopy(traveling_pop)
        best_individuals_per_every_migration.append(current_pop[:])
        current_pop.append(leader)
        migration_number += 1

    for j in range(pop_size):
        x_vals.append(current_pop[j].parameters[0])
        y_vals.append(current_pop[j].parameters[1])
        z_vals.append(current_pop[j].function_value)

        all_x_vals.append(x_vals)


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


(SOMA(2, 30, 7, 0.2, 2))
fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.scatter(x_leaders_vals, y_leaders_vals, z_leaders_vals, color='y', marker="x", linewidth=10)
ax.scatter(x_vals, y_vals, z_vals, color='g', marker="x", linewidth=7)
X = np.arange(-500, 500, 2)
Y = np.arange(-500, 500, 2)
X, Y = np.meshgrid(X, Y)
# R = ((X - 3) ** 2 + (Y - 3) ** 2)
arr = [X, Y]
R = (schwefel_function(arr))
surf = ax.plot_surface(X, Y, R, color='b',
                       linewidth=0, antialiased=True, alpha=0.3)
plt.show()
