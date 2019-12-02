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
        total_f_val += -i * np.sin(np.sqrt(np.abs(i)))
    return total_f_val


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


def set_attractiveness(actual_fly: FireFly, next_fly: FireFly, gamma, beta0, m):
    # brightness_at_zero_distance = np.exp(-gamma * pow(0, 2))
    brightness_at_zero_distance = 1.9
    r = calculate_distance_between_points(actual_fly.parameters, next_fly.parameters)
    beta = brightness_at_zero_distance * np.exp(-gamma * pow(r, m))
    return beta


def set_attractiveness_test(actual_fly: FireFly, next_fly: FireFly):
    r = calculate_distance_between_points(actual_fly.parameters, next_fly.parameters)
    psi = 1.0
    beta = 1.9
    test = beta * (1 / (psi + r))
    return beta * (1 / (psi + r))


def move_fire_fly(actual_fly: FireFly, more_attractive_fly: FireFly, calc_attractiveness, alpha):
    new_params = copy.deepcopy(actual_fly.parameters)
    for i in range(len(actual_fly.parameters)):
        new_params[i] = actual_fly.parameters[i] + 0.5* (
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
    for i in range(len(actual_fly.parameters)):
        # if np.random.random(1)[0] > 0.6:
        new_params = actual_fly.parameters[i] + (actual_fly.parameters[i] + random.randint(-15, 15))
        if new_params > 500:
            actual_fly.parameters[i] = actual_fly.parameters[i] - random.randint(0, 30)
        elif new_params < -500:
            actual_fly.parameters[i] = actual_fly.parameters[i] + random.randint(0, 30)
        else:
            actual_fly.parameters[i] = new_params


gamma = 0.5
alpha = 0.3
m = 2
number_of_iterations = 50
pop_size = 10
D = 2
history_of_pops = list()


def ffa(number_of_iterations, pop_size, gamma, alpha, m, D):
    new_generation = init_generation(D, pop_size)
    history_of_pops.append(new_generation)
    current_pop = copy.deepcopy(new_generation)
    brightest_fire_fly = min(current_pop, key=attrgetter('function_value'))
    for k in range(number_of_iterations):
        new_travel_gen = list()
        for i in range(len(current_pop)):
            changed = False
            # if current_pop[i] == brightest_fire_fly:
            #     random_move(current_pop[i], alpha)
            #     current_pop[i].function_value = schwefel_function(current_pop[i].parameters)
            #     new_travel_gen.append(current_pop[i])
            # else:
            for j in range(len(current_pop)):
                if current_pop[i].function_value > current_pop[j].function_value:
                    changed = True
                    att = set_attractiveness(current_pop[i], current_pop[j], gamma, 1, m)
                    # att = set_attractiveness_test(current_pop[i], current_pop[j])
                    current_pop[i].parameters = move_fire_fly(current_pop[i], current_pop[j], att, alpha)
                    current_pop[i].function_value = schwefel_function(current_pop[i].parameters)
            if not changed:
                random_move(current_pop[i], alpha)
                current_pop[i].function_value = schwefel_function(current_pop[i].parameters)
            new_travel_gen.append(current_pop[i])
        current_pop.clear()
        current_pop = new_travel_gen[:]
        current_pop = sorted(current_pop, key=attrgetter('function_value'))
        history_of_pops.append(current_pop[:])
        brightest_fire_fly = min(current_pop, key=attrgetter('function_value'))
        print("Ite: " + str(k) + " best firefly is ", brightest_fire_fly)


ffa(number_of_iterations, pop_size, gamma, alpha, m, D)
