import numpy as np
import matplotlib.pyplot as plt
from operator import attrgetter
from mpl_toolkits.mplot3d import Axes3D
import copy
import random
from matplotlib import cm, animation
import math

towns = list()
pheromone_table = np.ones((len(towns), len(towns)))


def init_pheromone_table(table_size):
    return np.ones((table_size, table_size))


def calculate_distance_between_all_towns(towns):
    distance_table = np.zeros((len(towns), len(towns)))
    for i in range(len(towns)):
        for j in range(len(towns)):
            if i != j:
                distance_table[i][j] = float(calculate_distance_between_towns(towns[i], towns[j]))
    return distance_table


def calculate_visibility_matrix_between_all_towns(towns):
    visibility_matrix = np.zeros((len(towns), len(towns)))
    for i in range(len(towns)):
        for j in range(len(towns)):
            if i != j:
                visibility_matrix[i][j] = 1.0 / float(calculate_distance_between_towns(towns[i], towns[j]))
    return visibility_matrix


def calculate_distance_between_towns(actual_town, town):
    x_distance = abs(actual_town.x - town.x)
    y_distance = abs(actual_town.y - town.y)
    return np.sqrt((pow(x_distance, 2) + pow(y_distance, 2)))


def calculate_trans_prob(current_town_index, next_town_index, alpha, beta, distance_table, pheromone_table,
                         unvisited_towns):
    numerator = (pow(pheromone_table[current_town_index][next_town_index], alpha) * pow(
        distance_table[current_town_index][next_town_index], beta))
    denominator = 0.0
    for i in range(len(unvisited_towns)):
        denominator += (pow(pheromone_table[current_town_index][unvisited_towns[i].index], alpha) * pow(
            distance_table[current_town_index][unvisited_towns[i].index], beta))
    return numerator / denominator


def calculate_tau(actual_city, visit_city, p, pheromone_table, actual_ant):
    first_half = (1 - p) * pheromone_table[actual_city.index][visit_city.index]
    second_half = 1 / actual_ant.path
    pheromone_table[actual_city.index][visit_city.index] = first_half + second_half
    return first_half + second_half


class Town:
    def __init__(self, x, y, i):
        self.index = i
        self.x = x
        self.y = y

    def __repr__(self):
        return "Town of index: " + str(self.index) + " with x on " + str(self.x) + " and y on " + str(self.y)


class Ant:
    def __init__(self, i, current_town):
        self.index = i
        self.current_town = current_town
        self.visited_towns = list()
        self.unvisited_towns = list()
        self.transition_prob = []
        self.path = 0.0

    def __repr__(self):
        return "Ant of index: " + str(self.index) + " with path  " + str(self.path)

    def travel_the_world(self, towns: list, alpha, beta, distance_table, pheromone_table):
        self.unvisited_towns = towns[:]
        self.unvisited_towns.remove(self.current_town)
        first_town = self.current_town
        # while len(self.unvisited_towns) != 0:
        one_step_prob = []
        one_step_town = []
        while len(self.unvisited_towns) != 0:
            for i in self.unvisited_towns:
                one_step_prob.append(
                    calculate_trans_prob(self.current_town.index, i.index, alpha, beta, distance_table, pheromone_table,
                                         self.unvisited_towns))
                one_step_town.append(i)
            sums = sum(one_step_prob)
            selection = np.random.choice(one_step_town, 1, p=one_step_prob)
            self.path += calculate_distance_between_towns(self.current_town, selection[0])
            self.visited_towns.append(selection[0])
            self.unvisited_towns.remove(selection[0])
            self.current_town = selection[0]
            one_step_prob = []
            one_step_town = []
        self.visited_towns.append(first_town)
        self.current_town = first_town
        self.path += calculate_distance_between_towns(self.current_town, first_town)


def read_file_dataset(file):
    f = open(file, "r")
    for x in f:
        towns.append(x)
    f.close()


def generate_towns():
    generated_towns = list()
    read_file_dataset("towns_locations_test_small")
    for i in range(len(towns)):
        town_string = towns[i].split()
        generated_town = Town(int(town_string[1]), int(town_string[2]), i)
        generated_towns.append(generated_town)
    return generated_towns


def aco(number_of_iterations, alpha, beta, distance_table, pheromone_table, towns, colony_size, number_of_top_ants):
    colonies = list()
    for j in range(number_of_iterations):
        new_colony = list()
        starting_town =towns[np.random.randint(0, len(towns))]
        for k in range(colony_size):
            new_ant = Ant(k, starting_town)
            new_ant.visited_towns.append(new_ant.current_town)
            new_ant.travel_the_world(towns, alpha, beta, distance_table, pheromone_table)
            new_colony.append(new_ant)
        colony_to_append = copy.deepcopy(new_colony)
        colonies.append(colony_to_append)
        new_colony.clear()
        colony_to_append = sorted(colony_to_append, key=attrgetter('path'))
        for i in range(number_of_top_ants):
            for k in range(len(colony_to_append[i].visited_towns)):
                if k + 1 != len(colony_to_append[i].visited_towns):
                    calculate_tau(colony_to_append[i].visited_towns[k], colony_to_append[i].visited_towns[k + 1], 0.5,
                                  pheromone_table, colony_to_append[i])
        print("ite: " + str(j) + " best ant " + str(min(colony_to_append, key=attrgetter('path'))))
        colony_to_append.clear()
    return colonies


print("help")
x_axis = list()
y_axis = list()

alpha = 1
beta = 2
colony_size = 10
number_of_iterations = 50
first_gen = generate_towns()
best_ants_number = 5
distance_table = calculate_visibility_matrix_between_all_towns(first_gen)
pheromone_table = init_pheromone_table(len(first_gen))

ac_result = aco(number_of_iterations, alpha, beta, distance_table, pheromone_table, first_gen, colony_size,
                best_ants_number)
print("done")
for i in first_gen:
    x_axis.append(i.x)
    y_axis.append(i.y)

plt.scatter(x_axis, y_axis, marker="o")
# l, = ax.plot(x_axis, y_axis, marker="o")

# ax.plot()
plt.show()
