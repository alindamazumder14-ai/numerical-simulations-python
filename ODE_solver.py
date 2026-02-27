import numpy as np

# -----------------------------
# Euler Method for ODE
# -----------------------------
def euler_method(f, t0, y0, h, n):
    t_values = [t0]
    y_values = [y0]

    for i in range(n):
        y0 = y0 + h * f(t0, y0)
        t0 = t0 + h

        t_values.append(t0)
        y_values.append(y0)

    return np.array(t_values), np.array(y_values)


# -----------------------------
# Runge-Kutta 4th Order Method
# -----------------------------
def rk4_method(f, t0, y0, h, n):
    t_values = [t0]
    y_values = [y0]

    for i in range(n):
        k1 = f(t0, y0)
        k2 = f(t0 + h/2, y0 + h*k1/2)
        k3 = f(t0 + h/2, y0 + h*k2/2)
        k4 = f(t0 + h, y0 + h*k3)

        y0 = y0 + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t0 = t0 + h

        t_values.append(t0)
        y_values.append(y0)

    return np.array(t_values), np.array(y_values)