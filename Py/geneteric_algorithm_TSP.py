import numpy as np, random, operator, pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter


class Town:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def calculate_distance_between_towns(self, town):
        x_distance = abs(self.x - town.x)
        y_distance = abs(self.y - town.y)
        return np.sqrt((pow(x_distance, 2) + pow(y_distance, 2)))

    def __repr__(self):
        return "Town with x on " + str(self.x) + " and y on " + str(self.y)


def generate_towns(amount_of_towns):
    generatedtowns = list()
    for i in range(amount_of_towns):
        gentown = np.random.normal(2, 8, size=[1, 2])
        generatedtowns.append(Town(abs(gentown[0, 0]), abs(gentown[0, 1])))
    return generatedtowns


def calculate_fitness(actual_population: list()):
    fitness = 0
    town_iterator = 1
    # print(len(actual_population))
    while town_iterator < len(actual_population):
        fitness += actual_population[town_iterator - 1].calculate_distance_between_towns(
            actual_population[town_iterator])
        town_iterator = town_iterator + 1

        fitness += actual_population[town_iterator - 1].calculate_distance_between_towns(actual_population[0])
    return fitness


def create_generation(population_zero: list, size_of_pop):
    list_of_generation = [[0 for x in range(2)] for y in range(size_of_pop)]
    for j in range(0, size_of_pop):
        # list_of_generation.append(random.sample(population_zero, 10))
        list_of_generation[j][0] = (random.sample(population_zero, size_of_pop))
        list_of_generation[j][1] = calculate_fitness(list_of_generation[j][0])
    list_of_generation = sorted(list_of_generation, key=itemgetter(1))
    return list_of_generation


def evaluate_elements(populations: list):
    for o in range(0, len(populations)):
        populations[o][1] = calculate_fitness(populations[o][0])
    sorted_populations = sorted(populations, key=itemgetter(1))
    print(sorted_populations[0][1])
    return sorted_populations


# def select_elements(actual_generation):


def breed(other_element, elite):
    child = []
    child_p1 = []
    child_p2 = []
    other_element_gen = int(random.random() * len(other_element))
    elite_gen = int(random.random() * len(elite))

    start_gene = min(other_element_gen, elite_gen)
    end_gene = max(other_element_gen, elite_gen)

    for i in range(start_gene, end_gene):
        child_p1.append(other_element[i])

    child_p2 = [item for item in elite if item not in child_p1]

    child = child_p1 + child_p2
    return child


def mutate(element):
    size_of_element = len(element)
    position_of_swap = int(random.random() * size_of_element)
    second_position_of_swap = int(random.random() * size_of_element)
    # print(element)

    tmp = element[position_of_swap]
    element[position_of_swap] = element[second_position_of_swap]
    element[second_position_of_swap] = tmp
    return element


def breed_population(actual_generation, size_of_pop):
    breeded_population = list()
    breeded_population.append(actual_generation[0][0])
    for k in range(1, size_of_pop):
        breeded_population.append(breed(actual_generation[k][0], actual_generation[0][0]))
    return breeded_population


def mutate_population(breeded_population, size_of_pop):
    mutated_population = [[0 for x in range(2)] for y in range(size_of_pop)]
    # mutated_population = list()
    mutated_population[0][0] = breeded_population[0]
    mutated_population[0][1] = calculate_fitness(breeded_population[0])
    for l in range(1, size_of_pop):
        mutated_population[l][0] = (mutate(breeded_population[l]))
        mutated_population[l][1] = (calculate_fitness(breeded_population[l]))
    return mutated_population


def create_next_generation(actual_generation, size_of_pop):
    new_actual_generation = evaluate_elements(actual_generation)
    breeded_population = breed_population(new_actual_generation, size_of_pop)
    mutated_population = mutate_population(breeded_population, size_of_pop)
    return mutated_population


def genetic_algorithm_loop(number_of_generations, number_of_towns, pop_size):
    generated_towns = (generate_towns(number_of_towns))
    pop = create_generation(generated_towns, pop_size)
    for generation in range(0, number_of_generations):
        pop = create_next_generation(pop, pop_size)


def set_graph_vals(best_actual_element):
    x_vals.clear()
    y_vals.clear()
    for it in best_actual_element:
        x_vals.append(it.x)
        y_vals.append(it.y)


x_vals = list()
y_vals = list()

genetic_algorithm_loop(500, 25, 25)
# for i in generated_towns:
#     x_vals.append(i.x)
#     print(i)
#     y_vals.append(i.y)

# actual_generation = create_generation(generated_towns)

# actual_fitness = (calculate_fitness(population_zero))

# fig, ax = plt.scatter(x_vals,y_vals)
# l, = ax.plot(x_vals, y_vals, marker="o")
# plt.show()
