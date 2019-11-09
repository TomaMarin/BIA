import numpy as np, random, operator, pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from matplotlib import animation

fig, ax = plt.subplots()
l, = ax.plot([], [], marker="o")

ax.plot()

best_combinations = list()


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
        list_of_generation[j][0] = (random.sample(population_zero, len(population_zero)))
        list_of_generation[j][1] = calculate_fitness(list_of_generation[j][0])
    list_of_generation = sorted(list_of_generation, key=itemgetter(1))
    return list_of_generation


def evaluate_elements(populations: list):
    for o in range(0, len(populations)):
        populations[o][1] = calculate_fitness(populations[o][0])
    sorted_populations = sorted(populations, key=itemgetter(1))
    # print(sorted_populations[0][1])
    best_combinations.append(sorted_populations[0])
    return sorted_populations


# def select_elements(actual_generation):

def roulette_selection(populations: list):
    avg = 0
    worst_element_fitness = populations[len(populations) - 1][1]
    for p in range(0, len(populations)):
        avg += (1 - (populations[p][1] / worst_element_fitness))
        # print(1 - (populations[p][1] / worst_element_fitness))
    # print("end")
    # print((avg / len(populations) - (1 - (populations[len(populations) - 1][1] / worst_element_fitness))))
    random_el = populations[len(populations) - 5][0]
    for v in range(1, len(populations)):
        ran_number = random.random() * ((1 - (populations[0][1] / worst_element_fitness)) + (
                    1 - (populations[len(populations) - 1][1] / worst_element_fitness)))
        succes_per = (1 - (populations[v][1] / worst_element_fitness))
        if succes_per > random.random():
            random_el = populations[v][0]
            break
    return random_el


def breed(other_element, another_element):
    child = []
    child_p1 = []
    child_p2 = []
    other_element_gen = int(random.random() * len(other_element))
    another_element_gen = int(random.random() * len(another_element))

    start_gene = min(other_element_gen, another_element_gen)
    end_gene = max(other_element_gen, another_element_gen)

    for i in range(start_gene, end_gene):
        child_p1.append(other_element[i])

    child_p2 = [item for item in another_element if item not in child_p1]

    child = child_p1 + child_p2
    reversered_twin = child_p2 + child_p1
    if calculate_fitness(child) > calculate_fitness(reversered_twin):
        return reversered_twin

    return child


def mutate(element, mutation_rate):
    size_of_element = len(element)
    position_of_swap = int(random.random() * size_of_element)
    second_position_of_swap = int(random.random() * size_of_element)
    # print(element)

    if mutation_rate > random.random():
        tmp = element[position_of_swap]
        element[position_of_swap] = element[second_position_of_swap]
        element[second_position_of_swap] = tmp
    return element


def breed_population(actual_generation, size_of_pop, elite_size):
    breeded_population = list()
    for el in range(0, elite_size):
        breeded_population.append(actual_generation[el][0])
    # breeded_population.append(actual_generation[1][0])
    for k in range(elite_size, size_of_pop):
        breeded_population.append(breed(roulette_selection(actual_generation), roulette_selection(actual_generation)))
    return breeded_population


def mutate_population(breeded_population, size_of_pop, mutation_rate):
    mutated_population = [[0 for x in range(2)] for y in range(size_of_pop)]
    # mutated_population = list()
    # mutated_population[0][0] = breeded_population[0]
    # mutated_population[0][1] = calculate_fitness(breeded_population[0])
    for l in range(0, size_of_pop):
        mutated_population[l][0] = (mutate(breeded_population[l], mutation_rate))
        mutated_population[l][1] = (calculate_fitness(breeded_population[l]))
    return mutated_population


def create_next_generation(actual_generation, size_of_pop, mutation_rate):
    new_actual_generation = evaluate_elements(actual_generation)
    breeded_population = breed_population(new_actual_generation, size_of_pop, 1)
    mutated_population = mutate_population(breeded_population, size_of_pop, mutation_rate)
    return mutated_population


def genetic_algorithm_loop(number_of_generations, number_of_towns, pop_size, mutation_rate):
    generated_towns = (generate_towns(number_of_towns))
    pop = create_generation(generated_towns, pop_size)
    for generation in range(0, number_of_generations):
        pop = create_next_generation(pop, pop_size, mutation_rate)


x_vals = list()
y_vals = list()


def animate(ite):
    if ite > len(best_combinations):
        ani.frame_seq = ani.new_frame_seq()
        ite = 0
    plt.title("fitness: " + str(best_combinations[ite][1]) + " iteration: " + str(ite))
    x_vals.clear()
    y_vals.clear()
    best_actual_element = best_combinations[ite][0]
    for it in best_actual_element:
        x_vals.append(it.x)
        y_vals.append(it.y)
    x_vals.append(best_actual_element[0].x)
    y_vals.append(best_actual_element[0].y)
    l.set_xdata(x_vals)
    l.set_ydata(y_vals)
    plt.xticks(range(int(min(x_vals) - 2), int(max(x_vals) + 2)))
    plt.yticks(range(int(min(y_vals) - 2), int(max(y_vals) + 2)))
    plt.draw()
    return l,


genetic_algorithm_loop(300, 20, 50, 0.25)


ani = animation.FuncAnimation(
    fig, animate, interval=250, repeat=False)

# plt.show()
