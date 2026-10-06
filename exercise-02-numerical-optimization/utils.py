import matplotlib.pyplot as plt
import numpy as np


def plot_nlp(optimum):
    x, y = np.meshgrid(np.linspace(-2, 1, 200), np.linspace(-1, 2, 200))
    cost = 0.5 * (x - 1) ** 2 + 50 * (y - x**2) ** 2 + 0.5 * x**2
    fig, ax = plt.subplots(figsize=(7, 5))
    contour = ax.contourf(x, y, np.log1p(cost), levels=25)
    fig.colorbar(contour, ax=ax, label="log(1 + f)")
    feasible_y = np.linspace(-1, 2, 300)
    ax.plot(-(1 - feasible_y) ** 2, feasible_y, "w", label="g = 0")
    ax.plot(*np.asarray(optimum).ravel(), "ro", label="Solution")
    ax.set(xlabel="x", ylabel="y", xlim=(-2, 1), ylim=(-1, 2))
    ax.legend()
    return fig


def plot_ocp(states, actions, dt):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    time = np.arange(len(states)) * dt
    axes[0].plot(time, states[:, 0])
    axes[0].set_ylabel("Angle")
    axes[1].plot(time, states[:, 1])
    axes[1].set_ylabel("Angular velocity")
    axes[2].step(time[:-1], actions, where="post")
    axes[2].set(xlabel="Time", ylabel="Torque", ylim=(-1.1, 1.1))
    fig.tight_layout()
    return fig
