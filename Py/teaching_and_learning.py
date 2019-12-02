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
        total_f_val += i * np.sin(np.sqrt(np.abs(i)))
    return - total_f_val


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


def find_median_positions(current_pop, D):
    median_student: Point
    averagess = list()
    averages = np.zeros((D, 1), dtype=float)
    for i in current_pop:
        for j in range(len(i.parameters)):
            averages[j][0] += i.parameters[j]
    for i in range(len(averages)):
        averages[i][0] = averages[i][0] / (len(current_pop))
        averagess.append(averages[i][0])
    median_student = Point(averagess, schwefel_function(averagess))
    return median_student


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
    better_student = copy.deepcopy(student)
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

    if student.function_value > random_student.function_value:
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
all_x_vals = list()
all_y_vals = list()
all_z_vals = list()


def teach_and_learn(iterations, pop_size, D):
    init_gen = init_generation(D, pop_size)
    current_pop = copy.deepcopy(init_gen)
    for i in range(len(current_pop)):
        all_x_vals.append(current_pop[i].parameters[0])
        all_y_vals.append(current_pop[i].parameters[1])
        all_z_vals.append(current_pop[i].function_value)
    all_iterations.append(init_gen)
    for i in range(iterations):
        # medians = find_median(current_pop)
        median = find_median_positions(current_pop, D)
        teacher = None
        teacher = min(current_pop, key=attrgetter('function_value'))
        print("ite ", i, teacher, "   median f val ", median.function_value)
        # print("ite ", i, median)
        taught_class = list()
        learnt_class = list()
        for j in range(len(current_pop)):
            # taught_class = [teaching_phase(i, teacher, median) for i in current_pop if (i != teacher)]
            if current_pop[j] != teacher:
                new_student = teaching_phase(current_pop[j], teacher, median)
                new_student.function_value = schwefel_function(new_student.parameters)
                taught_class.append(new_student)
        for k in range(len(taught_class)):
            ran_student = taught_class[random.randint(0, len(taught_class) - 1)]
            if taught_class[k] != ran_student:
                learnt_student = learning_phase(taught_class[k], ran_student)
            else:
                learnt_student = learning_phase(taught_class[k], taught_class[random.randint(0, len(taught_class) - 1)])
            learnt_student.function_value = schwefel_function(learnt_student.parameters)
            learnt_class.append(learnt_student)
        learnt_class.append(teacher)
        taught_and_learnt_class = copy.deepcopy(learnt_class)
        current_pop.clear()

        all_iterations.append(taught_and_learnt_class[:])
        current_pop = (taught_and_learnt_class[:])
        taught_and_learnt_class.clear()
    return current_pop


test = teach_and_learn(60, 15, 2)
fig = plt.figure()
ax = fig.gca(projection='3d')
v, = ax.plot(all_x_vals, all_y_vals, all_z_vals, linestyle="", color='r', markersize=12, marker="o", linewidth=12)

help_x = list()
help_y = list()
help_z = list()


def animate_vals(ite):
    # print(ite, " vals")
    plt.title("Migrace  number: " + str(ite))
    if ite >= len(all_iterations):
        help_x.clear()
        help_y.clear()
        help_z.clear()
        ani_vals.frame_seq = ani_vals.new_frame_seq()
    else:
        help_x.clear()
        help_y.clear()
        help_z.clear()
        for i in all_iterations[ite]:
            help_x.append(i.parameters[0])
            help_y.append(i.parameters[1])
            help_z.append(i.function_value)
        # ydata = y_leaders_vals[ite]
        # xdata = x_leaders_vals[ite]
        # zdata = z_leaders_vals[ite]
        # help_x.append(xdata)
        # help_y.append(ydata)
        # help_z.append(zdata)
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


X = np.arange(-500, 500, 5)
Y = np.arange(-500, 500, 5)
X, Y = np.meshgrid(X, Y)
# R = ((X - 3) ** 2 + (Y - 3) ** 2)
arr = [X, Y]
R = (schwefel_function(arr))
ani_vals = animation.FuncAnimation(fig, animate_vals, interval=300, init_func=init_vals,  repeat=True)
surf = ax.plot_surface(X, Y, R, color='b',
                       linewidth=0, antialiased=True, alpha=0.1)
print("done")
plt.show()