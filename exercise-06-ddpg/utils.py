from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def encode_gif(frames, duration=40):
    if not frames:
        return None
    images = [Image.fromarray(frame) for frame in frames]
    buffer = BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=duration, loop=0)
    return buffer.getvalue()


def plot_training(returns, q_losses, actor_losses):
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    axes[0].plot(returns, alpha=0.4)
    window = min(10, len(returns))
    if window:
        axes[0].plot(np.arange(window - 1, len(returns)),
                     np.convolve(returns, np.ones(window) / window, "valid"))
    axes[0].set(xlabel="Episode", ylabel="Return")
    axes[1].plot(q_losses)
    axes[1].set(xlabel="Gradient update", ylabel="Critic loss")
    axes[2].plot(actor_losses)
    axes[2].set(xlabel="Gradient update", ylabel="Actor loss (-Q)")
    fig.tight_layout()
    return fig


def plot_policy_value(actor, critic, n=101):
    theta, omega = np.linspace(-np.pi, np.pi, n), np.linspace(-8.0, 8.0, n)
    T, W = np.meshgrid(theta, omega, indexing="ij")
    states = torch.as_tensor(np.stack([np.cos(T), np.sin(T), W], axis=-1).reshape(-1, 3), dtype=torch.float32)
    with torch.no_grad():
        actions = actor(states)
        values = critic(states, actions)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    extent = [theta[0], theta[-1], omega[0], omega[-1]]
    for ax, data, title in zip(axes, [actions, values], ["Policy $\\mu_\\theta(s)$", "Value $Q_w(s,\\mu_\\theta(s))$"]):
        image = ax.imshow(data.reshape(n, n).numpy().T, origin="lower", aspect="auto", extent=extent)
        fig.colorbar(image, ax=ax)
        ax.set(title=title, xlabel="Angle", ylabel="Angular velocity")
    fig.tight_layout()
    return fig
