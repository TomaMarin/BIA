import plotly.offline
import plotly.graph_objs as go
import array
from random import randrange
import random
import math

import pylab
from matplotlib import animation


def find_global_min(arrayofx):
    pow_of_x = list()
    for element in arrayofx:
        pow_of_x.append(pow(element, 2))
    print("global minimum is at: ", math.sqrt(min(pow_of_x)), ", ", min(pow_of_x))
    return pow_of_x


def blind_search(smallest_number, biggest_number, iterations):
    random_list = list()
    for i in range(iterations):
        rand_int = random.randint(smallest_number, biggest_number)
        random_list.append(rand_int)
    return random_list


random_list = blind_search(0, 10, 10)
random_list.sort()
print(random_list)


# plotly.offline.plot(
#     dict(data=[go.Scatter(x=random_list, y=find_global_min(random_list))],
#          layout=go.Layout(title="hello world")),
#     image='jpeg', image_filename='test')


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=2, blit=True, save_count=50)
# pylab.plot(random_list, find_global_min(random_list))
pylab.show()
