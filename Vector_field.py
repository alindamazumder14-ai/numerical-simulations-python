import numpy as np
import matplotlib.pyplot as plt

def plot_vector_field(f):
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x, y)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(len(x)):
        for j in range(len(y)):
            dx, dy = f(X[i, j], Y[i, j])
            U[i, j] = dx
            V[i, j] = dy

    plt.figure()
    plt.quiver(X, Y, U, V)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Vector Field")
    plt.grid()
    plt.show()