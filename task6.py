"""
Решаем задачу 6.
Система единиц: СИ, но вместо секунд - часы
"""
import numpy as np
import matplotlib.pyplot as plt

def f(x:float,R:np.ndarray) -> np.ndarray:
    """
    Работает с вектором r = { x, x', y , y'}
    """
    # return some function result
    r = np.sqrt(R[0]**2+R[2]**2)
    return np.array([R[1], -q*M/r**3 * R[0], R[3], -q*M/r**3 * R[2]])
def dormand_prince(t_0,R_0,h,N):
    """
    https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method <- таблица Бутчера

    x_0: точка, где заданы функция и производная
    Y_0: {y(x_0), y'(x_0)}
    """
    t_n = t_0
    R_n = R_0.copy()
    tes, xes, yes = [],[],[]
    tes.append(t_n)
    xes.append(R_n[0])
    yes.append(R_n[2])
    for _ in range(N):
        if R_n[0]**2+R_n[2]**2<R**2:
            break
        k_1 = f(t_n,         R_n)
        k_2 = f(t_n+h/5,     R_n+h*k_1/5)
        k_3 = f(t_n+3*h/10,  R_n+h*k_1*3/40+h*k_2*9/40)
        k_4 = f(t_n+4/5*h,   R_n+44*h*k_1/55 - 56*h*k_2/15 + 32*h*k_3/9)
        k_5 = f(t_n+8/9*h,   R_n+19372*h*k_1/6561 - 25360/2187*h*k_2+ 64448/6561*h*k_3 - 212/729*h*k_4)
        k_6 = f(t_n+h,       R_n+9017/3168*k_1*h - 355/33*k_2*h + 46732/5247*k_3*h +49/176*k_4*h - 5103/18656*h*k_5)
        k_7 = f(t_n+h,       R_n+35/384*k_1*h +0+ 500/1113*k_3*h + 125/192*k_4*h-2187/6784*k_5*h + 11/84*h*k_6)
        # print(k_1, k_2, k_3, k_4, k_5, k_6, k_7)
        R_n += h*(35/384*k_1 + 500/1113*k_3 + 125/192*k_4 -2187/6784*k_5 + 11/84*k_6)
        t_n += h
        tes.append(t_n)
        xes.append(R_n[0])
        yes.append(R_n[2])
    return np.array(tes), xes, yes

M = 5.99e24
R = 6380e3
q = 6.67e-11
r_c = 10e7

"""
Вычисление динамики полета. Введем параметр "время" - t
"""
u = 0
v_c = np.sqrt(q*M/r_c)

T = 2*np.pi * r_c**(3/2)/np.sqrt(q*M)
h_1 = 5
N_1 = int(T/h_1)
h_2 = 10
N_2 = int(T/h_2)
h_3 = 20
N_3 = int(T/h_3)
t_0 = 0
R_0 = np.array([r_c, 0, 0, v_c - u])
tes_1, xes_1, yes_1 = dormand_prince(t_0,R_0,h_1,N_1)
tes_2, xes_2, yes_2 = dormand_prince(t_0,R_0,h_2,N_2)
tes_3, xes_3, yes_3 = dormand_prince(t_0,R_0,h_3,N_3)
xes_e = []
yes_e = []
delta_1 = np.sqrt((xes_1[0]-xes_1[len(xes_1)-1])**2 + (yes_1[0]-yes_1[len(yes_1)-1])**2)
delta_2 = np.sqrt((xes_2[0]-xes_2[len(xes_2)-1])**2 + (yes_2[0]-yes_2[len(yes_2)-1])**2)
delta_3 = np.sqrt((xes_3[0]-xes_3[len(xes_3)-1])**2 + (yes_3[0]-yes_3[len(yes_3)-1])**2)
print("Промах 1",delta_1)
print("Промах 2",delta_2)
print("Промах 3",delta_3)
print(delta_3/ delta_2 , delta_2/delta_1)
for t in range(6000):
    xes_e.append(R*np.cos(t/1000))
    yes_e.append(R*np.sin(t/1000))
plt.plot(xes_e, yes_e)

plt.scatter(xes_1, yes_1)
plt.show()
