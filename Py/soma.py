import numpy as np
import matplotlib.pyplot as plt
from operator import attrgetter
from mpl_toolkits.mplot3d import Axes3D


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


def SOMA(D, number_of_migrations, pop_size, PRT, min_div, path_length):
    init_pop = init(D, pop_size)
    migration_number = 0
    current_pop = init_pop
    while migration_number < number_of_migrations:
        leader = min(current_pop, key=attrgetter('function_value'))
        x_leaders_vals.append(leader.parameters[0])
        y_leaders_vals.append(leader.parameters[1])
        z_leaders_vals.append(leader.function_value)
        print("leader f val : ", leader.function_value)
        traveling_pop = list()
        traveling_pop.clear()
        for i in range(pop_size):
            if current_pop[i] != leader:
                individual_parameters = travel(D, PRT, path_length, current_pop[i], leader)
                traveling_pop.append(Individual(individual_parameters, custom_function(individual_parameters)))
        # print("travel pop", traveling_pop)
        current_pop.clear()
        current_pop = traveling_pop
        current_pop.append(leader)
        # print("curr pop",current_pop)
        migration_number += 1

    for j in range(pop_size):
        x_vals.append(current_pop[j].parameters[0])
        y_vals.append(current_pop[j].parameters[1])
        z_vals.append(current_pop[j].function_value)


best_individuals_per_every_migration = list()


def init(D, pop_size):
    current_pop = list()

    for i in range(pop_size):
        individual_params = np.random.normal(0, 6, size=[1, D])
        new_individual = Individual(individual_params[0], custom_function(individual_params[0]))
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
    leader_positions = leader.parameters
    actual_positions = individual.parameters
    start_positions = individual.parameters
    best_positions = start_positions
    t = 0.11
    step = 0.11
    while t < path_length:
        for i in range(D):
            actual_positions[i] = start_positions[i] + (leader_positions[i] - start_positions[i]) * t * ptv[i]
        if custom_function(actual_positions) <= custom_function(start_positions):
            best_positions = actual_positions
        t += step
    return best_positions


(SOMA(2, 45, 10, 0.2, 0.7, 3))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x_vals, y_vals, z_vals, marker="*", markersize=12, linewidth=3)
plt.show()
