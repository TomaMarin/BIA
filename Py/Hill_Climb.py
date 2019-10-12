import numpy as np
import plotly.offline
import plotly.graph_objs as go
import array
from random import randrange
import random
import math
import pylab
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button


def calculate_function_value(num):
    return pow((num - 3), 2)


def find_global_min(arrayofx):
    min_of_list = min(arrayofx, key=calculate_function_value)
    # print(min_of_list)
    return min_of_list


def generate_normal_distribution_list(best_previous_value, scatter_value, size_of_list):
    generated_list = (np.random.normal(best_previous_value, scatter_value, size_of_list))
    # print(generated_list)
    return generated_list


fitness = 0.5
values_difference = 0.5
amount_of_values = 10
number_of_iterations = 100

x_vals = list()
y_vals = list()

all_x_vals = list()
all_y_vals = list()

for i in range(number_of_iterations):
    gene_list = generate_normal_distribution_list(fitness, values_difference, amount_of_values)
    for j in gene_list:
        all_x_vals.append(j)
        all_y_vals.append(calculate_function_value(j))
        if calculate_function_value(fitness) > calculate_function_value(j):
            fitness = j
            x_vals.append(j)
            y_vals.append(calculate_function_value(j))
            print("true")

print(fitness)

# plotly.offline.plot(
#     dict(data=[go.Scatter(x=x_vals, y=y_vals)],
#          layout=go.Layout(title="fn(x) = (x-3)^2")),
#     image='jpeg', image_filename='test')
# l, = plt.plot(x_vals, y_vals, lw=2, marker="o")


fig, ax = plt.subplots()
l, = ax.plot(x_vals, y_vals, marker="o")

help_x = list()
help_y = list()


k = ax.plot(x_vals, y_vals)


class Index(object):
    ind = 0

    def next(self, event):
        print(self.ind, "+")

        if self.ind == len(x_vals):
            self.ind = 0
            ydata = y_vals[self.ind]
            xdata = x_vals[self.ind]
            help_x.clear()
            help_y.clear()
            help_x.append(xdata)
            help_y.append(ydata)
            print(help_x)
        else:
            ydata = y_vals[self.ind]
            xdata = x_vals[self.ind]
            help_x.append(xdata)
            help_y.append(ydata)

        l.set_ydata(help_y)
        l.set_xdata(help_x)
        # k = plt.scatter(help_x, help_y, lw=2)
        plt.draw()
        self.ind += 1

    def prev(self, event):
        print(self.ind, "-")

        if self.ind == 0:
            self.ind = len(x_vals)
            help_y.extend(y_vals)
            help_x.extend(x_vals)
            print(help_x)

        else:
            self.ind -= 1
            ydata = y_vals[self.ind]
            xdata = x_vals[self.ind]
            help_x.remove(xdata)
            help_y.remove(ydata)
        l.set_ydata(help_y)
        l.set_xdata(help_x)
        plt.draw()


print(len(y_vals))


def animate(ite):
    print(ite)
    if ite >= len(y_vals):
        help_x.clear()
        help_y.clear()
        ani.frame_seq = ani.new_frame_seq()

    else:
        ydata = y_vals[ite]
        xdata = x_vals[ite]
        help_x.append(xdata)
        help_y.append(ydata)
    l.set_ydata(help_y)
    l.set_xdata(help_x)
    return l,


def init():  # only required for blitting to give a clean slate.
    help_x.clear()
    help_y.clear()
    ydata = y_vals[0]
    xdata = x_vals[0]
    help_x.append(xdata)
    help_y.append(ydata)
    l.set_ydata(help_y)
    l.set_xdata(help_x)
    return l,


def start(self):
    ani.event_source.start()


def stop(self):
    ani.event_source.stop()


def clear(self):
    help_y.clear()
    help_x.clear()
    plt.draw()


ani = animation.FuncAnimation(
    fig, animate, interval=450, repeat=True)
# pylab.scatter(x_vals, y_vals)


callback = Index()
axprev = plt.axes([0.70, 0.9, 0.1, 0.075])
axnext = plt.axes([0.81, 0.9, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

axstart = plt.axes([0.55, 0.9, 0.1, 0.075])
axstop = plt.axes([0.45, 0.9, 0.1, 0.075])
axclear = plt.axes([0.35, 0.9, 0.1, 0.075])
clearB = Button(axclear, 'Clear')
clearB.on_clicked(clear)
startB = Button(axstart, 'Start')
startB.on_clicked(start)
stopB = Button(axstop, 'Pause')
stopB.on_clicked(stop)

plt.show()
