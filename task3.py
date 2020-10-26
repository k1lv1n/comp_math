import numpy as np


def f(x):
    return np.sin(100 * x) * np.exp(-x ** 2) * np.cos(2 * x)


def runge(I_1, I_2, p):
    return (I_1 - I_2) / (2 ** p - 1)


"""
Ососбых точек нет(как и информации о требуемой точности)
"""
n_1 = 20000
xes_1 = np.linspace(0, 3, n_1)
n_2 = 40000
xes_2 = np.linspace(0, 3, n_2)
"""
Метод левого прямоугольника
"""


def left_side_rectangle():
    p = 1
    sum_1 = 0
    for i in range(len(xes_1) - 1):
        sum_1 += f(xes_1[i]) * (xes_1[i + 1] - xes_1[i])
    sum_2 = 0
    for i in range(len(xes_2) - 1):
        sum_2 += f(xes_2[i]) * (xes_2[i + 1] - xes_2[i])
    err = runge(sum_1, sum_2, p)
    return sum_1 + err


"""
Метод правого прямоугольника
"""


def right_side_rectangle():
    p = 1
    sum_1 = 0
    for i in range(1, len(xes_1)):
        sum_1 += f(xes_1[i]) * (xes_1[i] - xes_1[i - 1])
    sum_2 = 0
    for i in range(1, len(xes_2)):
        sum_2 += f(xes_2[i]) * (xes_2[i] - xes_2[i - 1])
    err = runge(sum_1, sum_2, p)
    return sum_1 + err


"""
Метод средней точки
"""


def central_rectangle():
    p = 2
    sum_1 = 0
    for i in range(len(xes_1) - 1):
        sum_1 += f((xes_1[i] + xes_1[i + 1]) / 2) * (xes_1[i + 1] - xes_1[i])
    sum_2 = 0
    for i in range(len(xes_1) - 1):
        sum_2 += f((xes_2[i] + xes_2[i + 1]) / 2) * (xes_2[i + 1] - xes_2[i])
    err = runge(sum_1, sum_2, p)
    return sum_1 + err


def tracepies_method():
    p = 2
    sum_1 = 0
    for i in range(len(xes_1) - 1):
        sum_1 += ((f(xes_1[i]) + f(xes_1[i + 1])) / 2) * (xes_1[i + 1] - xes_1[i])
    sum_2 = 0
    for i in range(len(xes_1) - 1):
        sum_2 += ((f(xes_2[i]) + f(xes_2[i + 1])) / 2) * (xes_2[i + 1] - xes_2[i])
    err = runge(sum_1, sum_2, p)
    return sum_1 + err


def method_38():
    p = 4
    sum_1 = 0
    for i in range(len(xes_1) - 1):
        sum_1 += (f(xes_1[i]) + 3 * f((2 * xes_1[i] + xes_1[i + 1]) / 3) + 3 * f((xes_1[i] + 2 * xes_1[i + 1]) / 3) + f(
            xes_1[i + 1])) / 8 * (xes_1[i + 1] - xes_1[i])

    sum_2 = 0
    for i in range(len(xes_2) - 1):
        sum_2 += (f(xes_2[i]) + 3 * f((2 * xes_2[i] + xes_2[i + 1]) / 3) + 3 * f((xes_2[i] + 2 * xes_2[i + 1]) / 3) + f(
            xes_2[i + 1])) / 8 * (xes_2[i + 1] - xes_2[i])
    err = runge(sum_1, sum_2, p)
    # err = 0
    return sum_1 + err


def simpsons_method():
    p = 4
    sum_1 = 0
    for i in range(len(xes_1) - 1):
        sum_1 += (f(xes_1[i]) + 4 * f((xes_1[i] + xes_1[i + 1]) / 2) + f(xes_1[i + 1])) / 6 * (xes_1[i + 1] - xes_1[i])

    sum_2 = 0
    for i in range(len(xes_2) - 1):
        sum_2 += (f(xes_2[i]) + 4 * f((xes_2[i] + xes_2[i + 1]) / 2) + f(xes_2[i + 1])) / 6 * (xes_2[i + 1] - xes_2[i])
    err = runge(sum_1, sum_2, p)
    # err = 0
    return sum_1 + err


print(left_side_rectangle())
print(right_side_rectangle())
print(central_rectangle())
print(tracepies_method())
print(method_38())
print(simpsons_method())
