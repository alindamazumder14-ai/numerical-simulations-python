import numpy as np
import matplotlib.pyplot as plt
from ODE_solver import euler_method, rk4_method
from Vector_field import plot_vector_field

# -----------------------------
# Example 1: First Order ODE
# dy/dt = -2y
# Exact solution: y = e^(-2t)
# -----------------------------

def f(t, y):
    return -2*y

t0 = 0
y0 = 1
h = 0.1
n = 50

t_euler, y_euler = euler_method(f, t0, y0, h, n)
t_rk4, y_rk4 = rk4_method(f, t0, y0, h, n)

# Exact solution
t_exact = np.linspace(0, 5, 200)
y_exact = np.exp(-2*t_exact)

plt.figure()
plt.plot(t_euler, y_euler, label="Euler")
plt.plot(t_rk4, y_rk4, label="RK4")
plt.plot(t_exact, y_exact, label="Exact", linestyle="--")

plt.xlabel("t")
plt.ylabel("y")
plt.title("Numerical Solution of dy/dt = -2y")
plt.legend()
plt.grid()
plt.show()


# -----------------------------
# Example 2: Vector Field
# System:
# dx/dt = y
# dy/dt = -x
# -----------------------------

def system(x, y):
    return y, -x

plot_vector_field(system)