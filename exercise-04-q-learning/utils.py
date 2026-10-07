from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_training(rewards, lengths):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    window = min(20, len(rewards))
    for ax, data, label in zip(axes, [rewards, lengths], ["Episode return", "Episode length"]):
        ax.plot(data, alpha=0.3)
        ax.plot(np.arange(window - 1, len(data)), np.convolve(data, np.ones(window) / window, "valid"))
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Episode")
    fig.tight_layout()
    return fig


def plot_policy(Q):
    values = Q.max(axis=1).reshape(4, 12)
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(values)
    fig.colorbar(image, ax=ax, label="max Q(s,a)")
    arrows = ["↑", "→", "↓", "←"]
    for state in range(48):
        row, col = divmod(state, 12)
        label = "cliff" if row == 3 and 1 <= col <= 10 else arrows[Q[state].argmax()]
        if state == 47:
            label = "goal"
        ax.text(col, row, label, ha="center", va="center", color="white", fontsize=9)
    ax.set(title="Greedy policy and estimated value", xticks=[], yticks=[])
    return fig


def encode_gif(frames, duration=300):
    images = [Image.fromarray(frame) for frame in frames]
    buffer = BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=duration, loop=0)
    return buffer.getvalue()
