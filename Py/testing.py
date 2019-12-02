import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# fig, ax = plt.subplots()
#
# x = np.arange(0, 2*np.pi, 0.01)
# line, = ax.plot(x, np.sin(x))
#
#
# def init():  # only required for blitting to give a clean slate.
#     line.set_ydata([np.nan] * len(x))
#     return line,
#
#
# def animate(i):
#     line.set_ydata(np.sin(x + i / 100))  # update the data.
#     return line,
#
#
# ani = animation.FuncAnimation(
#     fig, animate, init_func=init, interval=2, blit=True, save_count=50)
#
# # To save the animation, use e.g.
# #
# # ani.save("movie.mp4")
# #
# # or
# #
# # from matplotlib.animation import FFMpegWriter
# # writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# # ani.save("movie.mp4", writer=writer)
#
# plt.show()

from multiprocessing import Pool
from itertools import starmap

import numpy

d = 5


def square(x):
    return x * x


d_list = list()
for i in range(5):
    d_list.append(square(numpy.random.randint(0, 10, size=5)))

# if __name__ == '__main__':
#     p = Pool(4)
#     result = p.starmap(square,  d_list[0])
#     print(result)
result = map(square, d_list)


# print(result)

def schwefel_function(x):
    total_f_val = 0.0
    for i in x:
        # f7(x)=sum(-x(i)·sin(sqrt(abs(x(i))))), i=1:n; -500<=x(i)<=500.
        total_f_val += i * np.sin(np.sqrt(np.abs(i)))
    return total_f_val


def schwefel_function2(x):
    total_f_val = 0.0
    for i in x:
        # f7(x)=sum(-x(i)·sin(sqrt(abs(x(i))))), i=1:n; -500<=x(i)<=500.
        total_f_val += i * np.sin(np.sqrt(np.abs(i)))
    return 418.9829 * 2 - total_f_val


def schaffer_n2(x):
    x1 = x[0]
    x2 = x[1]
    poweer = (pow(x1, 2) - pow(x2, 2))
    poweer2 = (pow(x1, 2) + pow(x2, 2))
    result = 0.5 + ((pow(np.sin(poweer), 2) - 0.5)
                    / (pow((1 + 0.001 * (poweer2)), 2))
                    )
    return result


print(schwefel_function2([404, -315]))
print(schaffer_n2([2.35, 4.3]))
print(schaffer_n2([2.12304, 4.2047]))
