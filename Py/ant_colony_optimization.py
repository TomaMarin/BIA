import numpy as np
import matplotlib.pyplot as plt
from operator import attrgetter
import copy
from multiprocessing import Pool
from timeit import default_timer as timer

towns = list()
pheromone_table = np.ones((len(towns), len(towns)))


def init_pheromone_table(table_size):
    return np.ones((table_size, table_size))


def calculate_distance_between_all_towns(towns):
    distance_table = np.zeros((len(towns), len(towns)))
    for i in range(len(towns)):
        for j in range(len(towns)):
            if i != j:
                distance_table[i][j] = (calculate_distance_between_towns(towns[i], towns[j]))
    return distance_table


def calculate_visibility_matrix_between_all_towns(towns):
    visibility_matrix = np.zeros((len(towns), len(towns)))
    for i in range(len(towns)):
        for j in range(len(towns)):
            if i != j:
                visibility_matrix[i][j] = 1.0 / float(calculate_distance_between_towns(towns[i], towns[j]))
    return visibility_matrix


def calculate_distance_between_towns(actual_town, town):
    # x_distance =
    # y_distance =
    return np.sqrt((pow(abs(actual_town.x - town.x), 2) + pow(abs(actual_town.y - town.y), 2)))


def calculate_trans_prob(current_town_index, next_town_index, alpha, beta, distance_table, pheromone_table,
                         denominator):
    numerator = (pow(pheromone_table[current_town_index][next_town_index], alpha) * pow(
        distance_table[current_town_index][next_town_index], beta))
    return numerator / denominator


def calculate_denominator_of_trans_prob_equation(current_town_index, alpha, beta, distance_table, pheromone_table,
                                                 unvisited_towns):
    denominator = 0.0
    for i in range(len(unvisited_towns)):
        denominator += (pow(pheromone_table[current_town_index][unvisited_towns[i].index], alpha) * pow(
            distance_table[current_town_index][unvisited_towns[i].index], beta))
    return denominator


def calculate_tau(actual_city, visit_city, pheromone_table, actual_ant):
    first_half = pheromone_table[actual_city.index][visit_city.index]
    second_half = 1.0 / actual_ant.path
    pheromone_table[actual_city.index][visit_city.index] = first_half + second_half
    return first_half + second_half


def evaporate(pheromone_table, p):
    evaporated_pheromone_table = copy.deepcopy(pheromone_table)
    for i in range(len(evaporated_pheromone_table)):
        for j in range(len(evaporated_pheromone_table)):
            evaporated_pheromone_table[i][j] = (1 - p) * evaporated_pheromone_table[i][j]
    return evaporated_pheromone_table


class MPAnt:
    def __init__(self, ant, towns, alpha, beta, distance_table, pheromone_table):
        self.ant = ant
        self.towns = towns
        self.alpha = alpha
        self.beta = beta
        self.distance_table = distance_table
        self.pheromone_table = pheromone_table


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
        one_step_prob = []
        one_step_town = []
        while len(self.unvisited_towns) != 0:
            act_denominator = calculate_denominator_of_trans_prob_equation(self.current_town.index, alpha, beta,
                                                                           distance_table, pheromone_table,
                                                                           self.unvisited_towns)
            one_step_prob = [
                calculate_trans_prob(self.current_town.index, i.index, alpha, beta, distance_table, pheromone_table,
                                     act_denominator) for i in self.unvisited_towns]
            selection = np.random.choice(self.unvisited_towns, 1, p=one_step_prob)
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
    read_file_dataset("towns_locations")
    for i in range(len(towns)):
        town_string = towns[i].split()
        generated_town = Town(int(town_string[1]), int(town_string[2]), i)
        generated_towns.append(generated_town)
    return generated_towns


def ant_init(new_ant, towns, alpha, beta, distance_table, pheromone_table):
    travel_ant = new_ant
    travel_ant.travel_the_world(towns, alpha, beta, distance_table, pheromone_table)
    return new_ant


the_best_ants = list()


def aco(number_of_iterations, alpha, beta, distance_table, pheromone_table, towns, colony_size, number_of_top_ants):
    colonies = list()
    best_ant = Ant(1500, towns[0])
    best_ant.path = 6000
    for j in range(number_of_iterations):
        new_colony = list()
        starting_town = towns[np.random.randint(0, len(towns))]
        for k in range(colony_size):
            new_ant = Ant(k, starting_town)
            new_ant.visited_towns.append(new_ant.current_town)
            new_ant.travel_the_world(towns, alpha, beta, distance_table, pheromone_table)
            new_colony.append(new_ant)
        colony_to_append = copy.deepcopy(new_colony)
        colonies.append(colony_to_append)
        new_colony.clear()

        colony_to_append = sorted(colony_to_append, key=attrgetter('path'))
        pheromone_table = evaporate(pheromone_table, 0.1)
        for i in range(number_of_top_ants):
            for k in range(len(colony_to_append[i].visited_towns)):
                if k + 1 < len(colony_to_append[i].visited_towns):
                    calculate_tau(colony_to_append[i].visited_towns[k], colony_to_append[i].visited_towns[k + 1],
                                  pheromone_table, colony_to_append[i])
        best_iteration_ant = min(colony_to_append, key=attrgetter('path'))

        if best_iteration_ant.path < best_ant.path and j > int(number_of_iterations / 5):
            best_ant = best_iteration_ant
            for k in range(len(best_ant.visited_towns)):
                if k + 1 < len(best_ant.visited_towns):
                    calculate_tau(best_ant.visited_towns[k], best_ant.visited_towns[k + 1],
                                  pheromone_table, best_ant)

        print("ite: " + str(j) + " best ant " + str(best_iteration_ant))
        colony_to_append.clear()
    the_best_ants.append(best_ant)
    print("The best found path was with ant: ", str(best_ant))
    return colonies


print("help")
x_axis = list()
y_axis = list()

alpha = 1.1
beta = 4.8
colony_size = 20
number_of_iterations = 300
first_gen = generate_towns()
best_ants_number = 20
distance_table = calculate_visibility_matrix_between_all_towns(first_gen)
pheromone_table = init_pheromone_table(len(first_gen))
start = timer()

ac_result = aco(number_of_iterations, alpha, beta, distance_table, pheromone_table, first_gen, colony_size,
                best_ants_number)
end = timer()
print("time :", end - start)
for i in the_best_ants[0].visited_towns:
    x_axis.append(i.x)
    y_axis.append(i.y)

fig, ax = plt.subplots()
l, = ax.plot(x_axis, y_axis, marker="o")

# ax.plot()
plt.show()
