import numpy as np
import matplotlib.pyplot as plt
from operator import attrgetter
from mpl_toolkits.mplot3d import Axes3D
import copy
import random
from matplotlib import cm, animation
import math
from statistics import mean


def calculate_distance_between_points(actual_point, point):
    # x_distance =
    # y_distance =
    result = 0
    for i in range(len(actual_point)):
        result += pow(abs(actual_point[i] - point[i]), 2)
    return np.sqrt(result)
    # return np.sqrt((pow(abs(actual_point.x - point.x), 2) + pow(abs(actual_point.y - point.y), 2)))


def schwefel_function(x):
    total_f_val = 0.0
    for i in x:
        # f7(x)=sum(-x(i)·sin(sqrt(abs(x(i))))), i=1:n; -500<=x(i)<=500.
        total_f_val += i * np.sin(np.sqrt(np.abs(i)))
    return -total_f_val


class FireFly:
    attractiveness = 0
    function_value = 0

    def __init__(self, parameters, attractiveness, function_value):
        self.parameters = parameters
        self.attractiveness = attractiveness
        self.function_value = function_value

    def __repr__(self):
        return "I am at " + str(self.parameters) + " with attractiveness " + str(
            self.attractiveness) + " with f val " + str(self.function_value)

    # def set_pbest(self, old_params):
    #     if schwefel_function(self.parameters) > schwefel_function(old_params):
    #         self.pBest = old_params[:]


def init_generation(D, pop_size):
    init_pop = list()
    for i in range(pop_size):
        actual_params = (random.sample(range(-500, 500), D))
        init_pop.append(
            FireFly(actual_params, 0, schwefel_function(actual_params)))
    return init_pop


def set_attractiveness(actual_fly: FireFly, next_fly: FireFly, gamma, m):
    brightness_at_zero_distance = np.exp(-gamma * pow(0, 2))
    # brightness_at_zero_distance = 1.9
    r = calculate_distance_between_points(actual_fly.parameters, next_fly.parameters)
    beta = brightness_at_zero_distance * np.exp(-gamma * pow(r, m))
    return beta


def set_attractiveness_test(actual_fly: FireFly, next_fly: FireFly):
    r = calculate_distance_between_points(actual_fly.parameters, next_fly.parameters)
    psi = 1.0
    return 1 / (psi + r)


def move_fire_fly(actual_fly: FireFly, more_attractive_fly: FireFly, calc_attractiveness, alpha):
    beta = 1.9
    new_params = copy.deepcopy(actual_fly.parameters)
    for i in range(len(actual_fly.parameters)):
        new_params[i] = actual_fly.parameters[i] + (beta * calc_attractiveness) * (
                more_attractive_fly.parameters[i] - actual_fly.parameters[i]) + alpha * (
                                np.random.uniform(size=1)[0] - 0.5)
        if new_params[i] < -500:
            new_params[i] = -500 + random.randint(0, 45)
        if new_params[i] > 500:
            new_params[i] = 500 - random.randint(0, 45)
        # actual_fly.parameters[i] = new_params[i]
    # if schwefel_function(new_params) < schwefel_function(actual_fly.parameters):
    return new_params
    # return actual_fly.parameters


def random_move(actual_fly: FireFly, alpha):
    new_params = copy.deepcopy(actual_fly.parameters)
    for i in range(len(actual_fly.parameters)):
        # if np.random.random(1)[0] > 0.6:
        new_params[i] = actual_fly.parameters[i] + alpha * (np.random.uniform(size=1)[0] - 0.5)
        if new_params[i] > 500:
            new_params[i] = actual_fly.parameters[i] - random.randint(0, 20)
        if new_params[i] < -500:
            new_params[i] = actual_fly.parameters[i] + random.randint(0, 20)

    # if schwefel_function(new_params) < schwefel_function(actual_fly.parameters):
    return new_params
    # return actual_fly.parameters


gamma = 0.5
alpha = 0.7
m = 2
number_of_iterations = 70
pop_size = 15
D = 2
history_of_pops = list()


def ffa(number_of_iterations, pop_size, gamma, alpha, m, D):
    new_generation = init_generation(D, pop_size)
    history_of_pops.append(new_generation)
    current_pop = copy.deepcopy(new_generation)
    current_pop = sorted(current_pop, key=attrgetter('function_value'))
    for k in range(number_of_iterations):
        new_travel_gen = list()
        for i in range(len(current_pop)):
            changed = False
            test_ff = copy.deepcopy(current_pop[i])
            for j in range(len(current_pop)):
                if current_pop[j].function_value < test_ff.function_value:
                    changed = True
                    att = set_attractiveness_test(test_ff, current_pop[j])
                    new_params = move_fire_fly(test_ff, current_pop[j], att, alpha)
                    new_ff = FireFly(new_params, 0, schwefel_function(new_params))
                    test_ff = copy.deepcopy(new_ff)
            if changed:
                new_travel_gen.append(test_ff)
            if not changed:
                params = random_move(current_pop[i], alpha)
                new_best_ff = FireFly(params, 0, schwefel_function(params))
                # current_pop[i] = new_best_ff
                new_travel_gen.append(new_best_ff)
        current_pop.clear()
        current_pop = new_travel_gen[:]
        current_pop = sorted(current_pop, key=attrgetter('function_value'))
        history_of_pops.append(current_pop[:])
        brightest_fire_fly = min(current_pop, key=attrgetter('function_value'))
        print("Ite: " + str(k) + " best firefly is ", brightest_fire_fly)


ffa(number_of_iterations, pop_size, gamma, alpha, m, D)
