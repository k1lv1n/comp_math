"""
Задание 4
"""
import numpy as np
"""
Вспомогательные функции
"""
def _poly_newton_coefficient(x, y):
    """
    Вычисление коэфициентов интерполяции в форме Ньютона
    """

    m = len(x)

    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])

    return a

def newton_polynomial(x_data, y_data, x):
    """
    Вычисление значения в точке по интерполяции
    """
    a = _poly_newton_coefficient(x_data, y_data)
    n = len(x_data) - 1 # Degree of polynomial
    p = a[n]

    for k in range(1, n + 1):
        p = a[n - k] + (x - x_data[n - k])*p

    return p
def tridiag_matrix_alg(A, F):
    """
    Решает трехдиагональную матрицу методом прогонки.
    """
    n = A.shape[0]
    alphas = np.zeros(shape=(n))
    bethas = np.zeros(shape =(n))
    alphas[1] = -A[0,1]/A[0,0]
    bethas[1] = F[0]/A[0,0]
    for i in range(1,n-1):
        a,b,c,f = A[i,i-1],A[i,i],A[i,i+1],F[i]
        alphas[i+1]=-c/(a*alphas[i]+b)
        bethas[i+1]=(f-a*bethas[i])/(a*alphas[i]+b)
    xes = np.zeros(shape = n)
    xes[n-1] = (F[n-1]-A[n-1,n-2]*bethas[-1])/(A[n-1,n-1]+A[n-1,n-2]*alphas[-1])
    for i in range(n-2,-1, -1):
        xes[i]=alphas[i+1]*xes[i+1]+bethas[i+1]
    return xes
"""
Вспомогательные функции для расчета(определяются краевой задачей)
"""
def g(x):
    return x**2-3
    # return x
def h(x):
    return (x**2-3)*np.cos(x)
    # return x
def f(x):
    return 2-6*x+2*x**3 +(x**2-3)*np.exp(x)*np.sin(x)*(1+np.cos(x))+np.cos(x)*(np.exp(x)+(x**2-1)+x**4-3*x**2)
    # return 2 - 6 * x + 2 * x * x * x + (x * x - 3) * np.exp(x) * np.sin(x) * (1 + np.cos(x)) + np.cos(x) * (np.exp(x) + (x * x - 1) + x * x * x * x - 3 * x * x)
def a(x,t):
    return 1/t**2 - g(x)/(2*t)
def b(x,t):
    return h(x) - 2/t**2
def c(x,t):
    return 1/t**2 + g(x)/(2*t)

def create_matrix(xes,t):
    """
    Создает прогоночную матрицу
    """
    N = len(xes)
    A = np.zeros(shape=(N,N))
    A[0,0],A[N-1,N-1]=1,1
    for i in range(1,N-1):
        A[i,i-1],A[i,i],A[i,i+1] = c(xes[i],t), b(xes[i],t), a(xes[i],t)
    return A
def create_values_vector(xes,bound_left,bound_right):
    """
    Создает вектор значений
    """
    N = len(xes)
    F = np.zeros(shape=(N))
    F[0],F[N-1] = bound_left, bound_right
    for i in range(1,N-1):
        F[i]=f(xes[i])
    return F



xes = np.linspace(0,3.1415926,25) # рассчетная сетка, последний аргумент - число узлов
t=xes[1]-xes[0] # шаг по времени. Определяется сеткой
A = create_matrix(xes,t) # прогоночная матрица
bound_left = 0 # граничное условие слева
bound_right = 3.1415926**2 # граничное условие справа
F = create_values_vector(xes,bound_left,bound_right) # вектор свободных членов
yes_solution = tridiag_matrix_alg(A,F) # решение на сетке, которое используется в интерполяции
required_xes = [0.5, 1, 1.5, 2, 2.5, 3]
required_yes = []
for x in required_xes:
    required_yes.append(newton_polynomial(xes,yes_solution,x))
print(required_yes)
draw_x = np.linspace(0,3.142,100)
draw_y = []
for x in draw_x:
    draw_y.append(newton_polynomial(xes,yes_solution,x))
import matplotlib.pyplot as plt
plt.plot(draw_x,draw_y)
plt.show()
