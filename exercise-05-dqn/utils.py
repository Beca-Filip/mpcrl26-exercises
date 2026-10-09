from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def encode_gif(frames, duration=40):
    if not frames:
        return None
    images = [Image.fromarray(frame) for frame in frames]
    buffer = BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=duration, loop=0)
    return buffer.getvalue()


def plot_training(returns, losses):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    axes[0].plot(returns, alpha=0.4)
    window = min(20, len(returns))
    axes[0].plot(np.arange(window - 1, len(returns)),
                 np.convolve(returns, np.ones(window) / window, "valid"))
    axes[0].set(xlabel="Episode", ylabel="Return", ylim=(0, 205))
    axes[1].plot(losses, alpha=0.5)
    axes[1].set(xlabel="Gradient update", ylabel="TD loss")
    fig.tight_layout()
    return fig
