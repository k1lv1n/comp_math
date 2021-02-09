"""
Файл с "солвером" уравнения переноса
"""

"""
ddt(u) + alpha *  ddx(u) = betha
"""
import numpy as np
import matplotlib.pyplot as plt

def boundary_func(t,x): # u(t,0)
    return 0
def initial_func(t,x): # u(0,x)
    return 0


alpha = 2
betha = 0

L,len_L=2,80 # длина отрезка по пространству и кол-во иттераций по пространству
T,len_T=10,2000 # длина отрезка по времени и кол-во отрезков по времени
x = np.linspace(0,L,len_L)
t = np.linspace(0,T,len_T)

t1,t2 = 0,0
t3=-1
for (index,t_ )in enumerate(t):
    if abs(t_-0.1)<1e-3:
        t1=index
    if abs(t_-1)<1e-3:
        t2 = index
        break

boundary_condition = boundary_func(t,0)
initial_condition = initial_func(0,x)
h,tau = x[1]-x[0], t[1]-t[0]

solution = np.zeros(shape=(len_T,len_L))
solution[0,:] = initial_condition
solution[:,0] = boundary_condition
solution[0, 0] = 1

class Explicit_scheme():
    """
    Используется явная схема (_|).
    """
    def get_next_point(self,n,m):
        """
        Вычисление и запись значения следующей точки в таблицу функции-решения
        """
        solution[n+1,m] = solution[n,m]+betha*tau - (tau*alpha/h)*(solution[n,m]-solution[n,m-1])

    def solve(self):
        """
        Решает уравнение
        """
        for n in range(len_T-1):
            for m in range(1,len_L):
                self.get_next_point(n,m)

expl = Explicit_scheme()
expl.solve()
# print(solution)
plt.scatter(x,solution[t1,:])
plt.scatter(x,solution[t1*2,:])
plt.scatter(x,solution[t1*3,:])
plt.scatter(x,solution[t2,:])
plt.scatter(x,solution[t2*2,:])
plt.scatter(x,solution[t3,:])
plt.show()
