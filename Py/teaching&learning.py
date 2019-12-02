import numpy as np
import matplotlib.pyplot as plt
from operator import attrgetter
from mpl_toolkits.mplot3d import Axes3D
import copy
import random
from matplotlib import cm, animation
import math


class Point:
    pBest = 0
    function_value = 0

    def __init__(self, parameters, function_value):
        self.parameters = parameters
        self.function_value = function_value

    def __repr__(self):
        return "I am at " + str(self.parameters) + " with f val " + str(self.function_value)

    def set_pbest(self, old_params):
        if schwefel_function(self.parameters) > schwefel_function(old_params):
            self.pBest = old_params[:]


def schwefel_function(x):
    total_f_val = 0.0
    for i in x:
        # f7(x)=sum(-x(i)·sin(sqrt(abs(x(i))))), i=1:n; -500<=x(i)<=500.
        total_f_val += -i * np.sin(np.sqrt(np.abs(i)))
    return total_f_val


def init_generation(D, pop_size):
    init_pop = list()
    for i in range(pop_size):
        actual_params = (random.sample(range(-500, 500), D))
        init_pop.append(
            Point(actual_params, schwefel_function(actual_params)))
    return init_pop


def find_median(population):
    total_f_val = 0.0
    for i in population:
        total_f_val += i.function_value
    median_number = total_f_val / len(population)
    median = min(population, key=lambda x: abs(x.function_value - median_number))
    return median


def teaching_phase(student: Point, teacher: Point, mean_value):
    better_student = copy.deepcopy(student)
    for i in range(len(student.parameters)):
        teaching_factor = round(1 + np.random.uniform(0, 1, 1)[0])
        value = better_student.parameters[i] + np.random.uniform(0, 1, 1)[0] * (
                teacher.parameters[i] - teaching_factor * mean_value.parameters[i])
        if value < -500:
            better_student.parameters[i] = -500 + random.randint(0, 50)
        elif value > 500:
            better_student.parameters[i] = 500 - random.randint(0, 50)
        else:
            better_student.parameters[i] = value
    if schwefel_function(better_student.parameters) < schwefel_function(student.parameters):
        better_student.function_value = schwefel_function(better_student.parameters)
        return better_student
    return student


def learning_phase(student: Point, random_student: Point):
    # teaching_factor = round(1 + np.random.uniform(0, 1, 1)[0])
    better_student = student
    if student.function_value <= random_student.function_value:
        for i in range(len(student.parameters)):
            value = better_student.parameters[i] + np.random.uniform(0, 1, 1)[0] * (
                    better_student.parameters[i] - random_student.parameters[i])
            if value < -500:
                better_student.parameters[i] = -500 + random.randint(0, 30)
            elif value > 500:
                better_student.parameters[i] = 500 - random.randint(0, 30)
            else:
                better_student.parameters[i] = value

    elif student.function_value > random_student.function_value:
        for i in range(len(student.parameters)):
            value = better_student.parameters[i] + np.random.uniform(0, 1, 1)[0] * (
                    random_student.parameters[i] - better_student.parameters[i])
            if value < -500:
                better_student.parameters[i] = -500 + random.randint(0, 30)
            elif value > 500:
                better_student.parameters[i] = 500 - random.randint(0, 30)
            else:
                better_student.parameters[i] = value

    if schwefel_function(better_student.parameters) < schwefel_function(student.parameters):
        better_student.function_value = schwefel_function(better_student.parameters)
        return better_student
    return student


all_iterations = list()


def teach_and_learn(iterations, pop_size, D):
    init_gen = init_generation(D, pop_size)
    current_pop = copy.deepcopy(init_gen)
    all_iterations.append(init_gen)
    for i in range(iterations):
        median = find_median(current_pop)
        teacher = None
        teacher = min(current_pop, key=attrgetter('function_value'))
        print("ite ", i, teacher,"   median f val ", median.function_value)
        # print("ite ", i, median)
        taught_class = list()
        learnt_class = list()
        for j in range(len(current_pop)):
            # taught_class = [teaching_phase(i, teacher, median) for i in current_pop if (i != teacher)]
            if current_pop[j] != teacher:
                new_student = teaching_phase(current_pop[j], teacher, median)
                taught_class.append(new_student)
        for k in range(len(taught_class)):
            learnt_student = learning_phase(taught_class[k], taught_class[random.randint(0, len(taught_class) - 1)])
            learnt_class.append(learnt_student)
        learnt_class.append(teacher)
        taught_and_learnt_class = copy.deepcopy(learnt_class)
        current_pop.clear()

        all_iterations.append(taught_and_learnt_class[:])
        current_pop = (taught_and_learnt_class[:])
        taught_and_learnt_class.clear()
    return current_pop


test = teach_and_learn(50, 15, 2)
print("done")
