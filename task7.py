"""
Решение задачи 7.
"""
import numpy as np
import matplotlib.pyplot as plt

def f(x:float,R:np.ndarray) -> np.ndarray:
    """
    Работает с вектором r = { x, x', y , y'}
    """
    # return some function result
    r_1 = np.sqrt((R[0]+mu)**2 + R[2]**2)
    r_2 = np.sqrt((R[0]-mu_)**2 + R[2]**2)
    tmp_1 = 2*R[3]+R[0]-mu_*(R[0]+mu)/r_1**3 - mu*(R[0]-mu_)/r_2**3 - k*R[1]
    tmp_3 = -2*R[1]+R[2]-mu_*R[2]/r_1**3 - mu*R[2]/r_2**3 - k*R[3]
    return np.array([R[1],tmp_1,R[3],tmp_3])
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

mu = 1/82.45
mu_ = 1 - mu
 # надо подобрать

t_0 = 0
T = 8
h=0.0005
N = int(T/h)
k=0
r_min = np.inf
res=0
# for i in range(-100000,100000):
    # R_0 = np.array([1.2, 0 , -1.05, i/1000])
    # tes, xes, yes = dormand_prince(t_0, R_0,h,N)
    # r = np.sqrt((xes[0]-xes[len(xes)-1])**2 + (yes[0]-yes[len(yes)-1])**2)
    # if r<r_min:
        # r_min = r
        # res = i/10000
        # print(i/10000)
res = -0.1264
R_0 = np.array([1.2, 0 , -1.05, -1.05])
# R_1 = np.array([1.2, 0 , -1.05, -1.1])
# R_2 = np.array([1.2, 0 , -1.05, -1.2])
k =0
tes, xes, yes = dormand_prince(t_0, R_0,h,N)
k=0.1
tes, xes_1, yes_1 = dormand_prince(t_0, R_0,h,N)
k=1
tes, xes_2, yes_2 = dormand_prince(t_0, R_0,h,N)
# plt.scatter(xes_2[0],yes_2[0],s=200)
plt.scatter(xes,yes,s=5)
plt.scatter(xes_1,yes_1,s=5,c = 'r')
plt.scatter(xes_2,yes_2,s=5,c = 'b')
plt.show()
