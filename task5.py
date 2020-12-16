import numpy as np
import matplotlib.pyplot as plt


def f(x:float,y:np.ndarray) -> np.ndarray:
    """
    Работает с вектором { y , y'}
    """
    # return some function result

    return np.array([y[1], np.sqrt(abs(-np.exp(y[1])*y[0] + 2.71*y[0]**2/np.log(x)+1/x**2))])
    # return np.array([y[1], -y[0]])
def dormand_prince(x_0,Y_0,h,N):
    """
    https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method <- таблица Бутчера

    x_0: точка, где заданы функция и производная
    Y_0: {y(x_0), y'(x_0)}
    """
    x_n = x_0
    Y_n = Y_0.copy()
    xes, yes = [],[]
    xes.append(x_n)
    yes.append(Y_n[0])
    for _ in range(int(N)):
        k_1 = f(x_n,         Y_n)
        k_2 = f(x_n+h/5,     Y_n+h*k_1/5)
        k_3 = f(x_n+3*h/10,  Y_n+h*k_1*3/40+h*k_2*9/40)
        k_4 = f(x_n+4/5*h,   Y_n+44*h*k_1/55 - 56*h*k_2/15 + 32*h*k_3/9)
        k_5 = f(x_n+8/9*h,   Y_n+19372*h*k_1/6561 - 25360/2187*h*k_2+ 64448/6561*h*k_3 - 212/729*h*k_4)
        k_6 = f(x_n+h,       Y_n+9017/3168*k_1*h - 355/33*k_2*h + 46732/5247*k_3*h +49/176*k_4*h - 5103/18656*h*k_5)
        k_7 = f(x_n+h,       Y_n+35/384*k_1*h +0+ 500/1113*k_3*h + 125/192*k_4*h-2187/6784*k_5*h + 11/84*h*k_6)
        # print(k_1, k_2, k_3, k_4, k_5, k_6, k_7)
        Y_n += h*(35/384*k_1 + 500/1113*k_3 + 125/192*k_4 -2187/6784*k_5 + 11/84*k_6)
        x_n += h
        xes.append(x_n)
        yes.append(Y_n[0])
    return np.array(xes), yes

x_0 = 2.71
Y_0 = np.array([2.71, 2.009],dtype = float) # функция и производная в точке х_0
"""
Из-за особенностей заданя, не представляется возмоность увеличить значение производной в начальной точке, поэтому 2
Так же, не стоит менять шаг, иначе все перестает работать ¯\_(ツ)_/¯
"""
L = [2.71, 7.34]
h_1 = 0.03
N_1 = (L[1]-L[0])/h_1
h_2 = 0.0005
N_2 = (L[1]-L[0])/h_2
# N = 100

xes_1 , yes_1 = dormand_prince(x_0,Y_0,h_2,N_2)
plt.scatter(xes_1, yes_1)
"""
Осталось задать значения функции в требуемых точках
"""
x_0 = 2.71
Y_0 = np.array([2.71, 2.009],dtype = float)
L_3 = [0.49, 2.71]
h_3 = -0.005
N_3 = (L_3[0]-L_3[1])/h_3

xes_2, yes_2 = dormand_prince(x_0, Y_0, h_3, N_3)
for i,x in enumerate(xes_2):
    if abs(x-0.5)<1e-3:
        print(x,yes_2[i])
    if abs(x-1)<1e-3:
        print(x,yes_2[i])
    if abs(x-1.5)<1e-3:
        print(x,yes_2[i])
    if abs(x-2)<1e-3:
        print(x,yes_2[i])
    if abs(x-2.5)<1e-3:
        print(x,yes_2[i])
plt.scatter(xes_2, yes_2)
plt.show()
