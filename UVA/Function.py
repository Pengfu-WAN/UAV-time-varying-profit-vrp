import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import fsolve

def value_decay(x, alpha, beta):
    return beta/pow((np.exp(x) + 1), alpha)

def information_value(x, a, b):
    return ((np.exp(-x + a) - np.exp(x - a))/(np.exp(x - a) + np.exp(-x + a)) + 1) * b / 2

def information_value_cum(func, service_start_time, service_end_time, a, b):
    return integrate.quad(func, service_start_time, service_end_time, args=(a, b))

def func_T(T, func, a, b):
    return [func(T, a, b)[0]]

def func_t(t, func1, func2, func3, a1, a2, b1, b2, alpha2, beta2, start_time, T1, T2, travel_time):
    b3 = func1(start_time + t + travel_time, alpha2, beta2)
    T3 = fsolve(func_T, a2, args=(func2, a2, b3))[0]
    return [func3(func2, t, T1, a1, b1)[0] + func3(func2, 0, T2, a2, b2)[0] - func3(func2, 0, T3, a2, b3)[0]]

def calculate_final_t(func1, func2, func3, alpha, beta, a, start_time, left_time):
    b = func1(start_time, alpha, beta)
    T = fsolve(func_T, a, args=(func2, a, b))[0]
    T = min(T, left_time)
    value = func3(func2, 0, T, a, b)
    return T, value

def calculate_t(func1, func2, func3, a1, a2, alpha1, alpha2, beta1, beta2, start_time, left_time, travel_time):
    b1 = func1(start_time, alpha1, beta1)
    T1 = fsolve(func_T, a1, args=(func2, a1, b1))[0]
    ##print("T1", T1)
    b2 = func1(start_time + T1 + travel_time, alpha2, beta2)
    T2 = fsolve(func_T, a2, args=(func2, a2, b2))[0]
    ##print("T2", T2)
    T2 = min(T2, left_time)
    # judge if condition exists
    b3 = func1(start_time + travel_time, alpha2, beta2)
    T3 = fsolve(func_T, a2, args=(func2, a2, b3))[0]
    if func3(func2, 0, T1, a1, b1)[0] + func3(func2, 0, T2, a2, b2)[0] - func3(func2, 0, T3, a2, b3)[0] < 0:
        return 0, [0]
    t = fsolve(func_t, 0, args=(func1, func2, func3, a1, a2, b1, b2, alpha2, beta2, start_time, T1, T2, travel_time))[0]
    ##print(ans)
    Value = func3(func2, 0, t, a1, b1)
    return t, Value



"""
x = 10
f = np.vectorize(value_decay)
y = f(x, 0.1)
print(y)

x2 = np.arange(1, 20)
f2 = np.vectorize(information_value)
y2 = f2(x2, 5, y)
fArea, err = information_value_cum(information_value, 20, 5, y)
print(fArea)
plt.plot(x2, y2)
plt.show()
"""
if __name__ == '__main__':
    a, b = calculate_t(value_decay, information_value, information_value_cum, 7, 8, 0.2, 0.01, 1, 2, 0, 100, 2)
    print(a, b)


