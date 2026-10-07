import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(grids, dp_value, dp_action, lqr_value, lqr_action):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    extent = [grids[0][0], grids[0][-1], grids[1][0], grids[1][-1]]
    for ax, data, title in zip(axes.ravel(), [dp_value, lqr_value, dp_action, lqr_action],
                               ["Value iteration cost-to-go", "LQR quadratic value", "Value iteration policy", "Clipped LQR policy"]):
        shown = np.where(data >= 1e11, np.nan, data)
        image = ax.imshow(shown.T, origin="lower", aspect="auto", extent=extent)
        fig.colorbar(image, ax=ax)
        ax.set(title=title, xlabel="Angle", ylabel="Angular velocity")
    fig.tight_layout()
    return fig


def plot_slices(grids, dp_value, lqr_value):
    fig, ax = plt.subplots(figsize=(8, 4))
    for j in [15, 25, 35]:
        ax.plot(grids[0], np.where(dp_value[:, j] < 1e11, dp_value[:, j], np.nan),
                label=f"Value iteration, omega={grids[1][j]:.1f}")
        ax.plot(grids[0], lqr_value[:, j], "--", label=f"LQR, omega={grids[1][j]:.1f}")
    ax.set(xlabel="Angle", ylabel="Cost-to-go")
    ax.legend()
    return fig
