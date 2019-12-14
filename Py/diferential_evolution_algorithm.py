import numpy as np, random, operator, pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from matplotlib import animation
from operator import attrgetter
import copy
from mpl_toolkits.mplot3d import Axes3D

def schwefel_function(x):
    total_f_val = 0.0
    for i in x:
        # f7(x)=sum(-x(i)·sin(sqrt(abs(x(i))))), i=1:n; -500<=x(i)<=500.
        total_f_val += -i * np.sin(np.sqrt(np.abs(i)))
    return total_f_val


class Individual:
    function_value = 0
    parameters = list()

    def __init__(self, parameters, function_value):
        self.parameters = parameters
        self.function_value = function_value

    def __repr__(self):
        return "I am at " + str(self.parameters) + " with f val " + str(self.function_value)


def init_generation(D, pop_size):
    init_pop = list()
    for i in range(pop_size):
        actual_params = (random.sample(range(-500, 500), D))
        init_pop.append(
            Individual(actual_params, schwefel_function(actual_params)))
    return init_pop


def mutate(a: Individual, b: Individual, c: Individual, f_mutation_constant):
    noise_vector = copy.deepcopy(c)
    for i in range(len(noise_vector.parameters)):
        noise_vector.parameters[i] = c.parameters[i] + f_mutation_constant * (
                a.parameters[i] - b.parameters[i])
        if noise_vector.parameters[i] > 500:
            noise_vector.parameters[i] = 500 - random.randint(0, 40)
        if noise_vector.parameters[i] < -500:
            noise_vector.parameters[i] = -500 + random.randint(0, 40)
    noise_vector.function_value = schwefel_function(noise_vector.parameters)
    return noise_vector


def crossover(parent: Individual, mutant: Individual, cross_over_rate):
    child = copy.deepcopy(mutant)
    for i in range(len(child.parameters)):
        if random.randint(0, len(child.parameters)) < cross_over_rate:
            child.parameters[i] = mutant.parameters[i]
        else:
            child.parameters[i] = parent.parameters[i]
    child.function_value = schwefel_function(child.parameters)
    return child


x_vals = list()
y_vals = list()
z_vals = list()
all_x_vals = list()
all_y_vals = list()
all_z_vals = list()

best_individuals_per_every_migration = list(list())


def dea(iterations, pop_size, cross_over_rate, f_mutation_constant, D):
    init_pop = init_generation(D, pop_size)
    best_individuals_per_every_migration.append(init_pop)
    for i in init_pop:
        all_x_vals.append(i.parameters[0])
        all_y_vals.append(i.parameters[1])
        all_z_vals.append(i.function_value)
    current_pop = init_pop[:]
    for i in range(iterations):
        new_pop = list()
        for j in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != j]
            random_idx = np.random.choice(idxs, 3, replace=False)
            a = current_pop[random_idx[0]]
            b = current_pop[random_idx[1]]
            c = current_pop[random_idx[2]]
            mutant = mutate(a, b, c, f_mutation_constant)
            cross_over_individual = crossover(current_pop[j], mutant, cross_over_rate)
            if cross_over_individual.function_value < current_pop[j].function_value:
                current_pop[j] = copy.deepcopy(cross_over_individual)
            new_pop.append(current_pop[j])
        current_pop.clear()
        current_pop = new_pop[:]
        best_individuals_per_every_migration.append(current_pop[:])
        print("best individual for ite ", i, "was ", min(current_pop, key=attrgetter('function_value')))
        for j in range(pop_size):
            x_vals.append(current_pop[j].parameters[0])
            y_vals.append(current_pop[j].parameters[1])
            z_vals.append(current_pop[j].function_value)


fig = plt.figure()
ax = fig.gca(projection='3d')
help_x = list()
help_y = list()
help_z = list()
v, = ax.plot(all_x_vals, all_y_vals, all_z_vals, linestyle="", color='r', markersize=12, marker="o", linewidth=12)


def animate_vals(ite):
    # print(ite, " vals")
    plt.title("Iteration  number: " + str(ite))
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



cross_over_rate = 0.8
f_mutation_constant = 0.6
D = 2
dea(40, 10, cross_over_rate, f_mutation_constant, D)


X = np.arange(-500, 500, 5)
Y = np.arange(-500, 500, 5)
X, Y = np.meshgrid(X, Y)
# R = ((X - 3) ** 2 + (Y - 3) ** 2)
arr = [X, Y]
R = (schwefel_function(arr))
ani_vals = animation.FuncAnimation(fig, animate_vals, interval=300, init_func=init_vals,  repeat=True)
surf = ax.plot_surface(X, Y, R, color='b',
                       linewidth=0, antialiased=True, alpha=0.1)
plt.show()
