from io import BytesIO

import casadi as ca
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from acados_template import plot_trajectories
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback

# Exercise 07 counts the steps inside this box around upright: |x|, |theta|, |v|, |omega|.
BOX = np.array([0.1, 0.02, 0.1, 0.02])



# ---------------------------------------------------------------- model


def cartpole_ode(x, u, M=1.0, m=0.1, g=9.81, l=0.8):
    """Cart pole from Exercise 07, state (x, theta, v, omega) with theta = 0 upright."""
    theta, v, omega, F = x[1], x[2], x[3], u[0]
    c, s = ca.cos(theta), ca.sin(theta)
    den = M + m - m * c**2
    return ca.vertcat(
        v,
        omega,
        (-m * l * s * omega**2 + m * g * c * s + F) / den,
        (-m * l * c * s * omega**2 + F * c + (M + m) * g * s) / (l * den),
    )


def rk4(f, x, u, h):
    k1 = f(x, u)
    k2 = f(x + h / 2 * k1, u)
    k3 = f(x + h / 2 * k2, u)
    k4 = f(x + h * k3, u)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------- environment


def in_box(obs):
    return np.all(np.abs(obs) < BOX, axis=-1).astype(np.float32)


class CartPoleEnv(gym.Env):
    """Swing-up from Exercise 07: start hanging, 80 steps of 0.05 s.

    With random_start, half of the episodes start near upright and the other half at a
    random angle, both with small position and speeds.
    The reward is 1 for every step inside BOX (the score of Exercise 07) plus
    (1 + cos theta) / 2 * exp(-(x / 0.5)^2), which is 0 hanging and 1 upright in the
    middle of the track and guides the swing-up. The episode terminates if the cart leaves the track.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, dt=0.05, max_steps=80, Fmax=80.0, x_threshold=2.4, render_mode=None, random_start=False):
        x, u = ca.SX.sym("x", 4), ca.SX.sym("u", 1)
        self.f = ca.Function("f", [x, u], [rk4(cartpole_ode, x, u, dt)])
        self.max_steps, self.Fmax, self.x_threshold = max_steps, Fmax, x_threshold
        self.render_mode, self.random_start = render_mode, random_start
        self.action_space = gym.spaces.Box(-Fmax, Fmax, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.plan = None  # optional planned states, drawn by render()
        self._fig = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start and self.np_random.random() < 0.5:
            self.s = self.np_random.uniform(-0.3, 0.3, size=4)  # near upright
        elif self.random_start:
            self.s = self.np_random.uniform([-0.5, -np.pi, -0.5, -0.5], [0.5, np.pi, 0.5, 0.5])
        else:
            self.s = np.array([0.0, np.pi, 0.0, 0.0])
        self.k = 0
        return self.features(self.s), {}

    def step(self, action):
        self.s = np.array(self.f(self.s, np.clip(action, -self.Fmax, self.Fmax))).flatten()
        self.s[1] = (self.s[1] + np.pi) % (2 * np.pi) - np.pi
        self.k += 1
        reward = float(in_box(self.s)) + 0.5 * (1 + np.cos(self.s[1])) * np.exp(-((self.s[0] / 0.5) ** 2))
        terminated = bool(abs(self.s[0]) > self.x_threshold)
        truncated = self.k >= self.max_steps
        return self.features(self.s), reward, terminated, truncated, {}

    @staticmethod
    def features(state):
        # The observation is (x, sin theta, cos theta, v, omega), without a jump at theta = +-pi.
        # The full state stays in self.s.
        x, theta, v, omega = state
        return np.array([x, np.sin(theta), np.cos(theta), v, omega], dtype=np.float32)

    def render(self):
        return self.draw(self.s, self.plan)

    def draw(self, state, plan=None, l=0.8):
        """Image of the cart pole in state, with the planned path of the pole tip."""
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 2.6), dpi=80)
        ax, (x, theta) = self._ax, state[:2]
        ax.clear()
        ax.axhline(0, color="k", lw=1, zorder=0)
        for edge in (-self.x_threshold, self.x_threshold):
            ax.axvline(edge, color="0.7", ls=":", zorder=0)
        if plan is not None:
            ax.plot(plan[:, 0] - l * np.sin(plan[:, 1]), l * np.cos(plan[:, 1]),
                    "o-", color="0.6", ms=3, lw=1, zorder=1)
        ax.add_patch(plt.Rectangle((x - 0.25, -0.12), 0.5, 0.24, color="0.15", zorder=2))
        ax.plot([x, x - l * np.sin(theta)], [0, l * np.cos(theta)], color="#ca9865", lw=6,
                solid_capstyle="round", zorder=3)
        ax.plot([x], [0], "o", color="#8184cb", ms=6, zorder=4)
        ax.set(xlim=(-2.6, 2.6), ylim=(-1.0, 1.0), aspect="equal", xticks=[], yticks=[])
        self._fig.canvas.draw()
        return np.asarray(self._fig.canvas.buffer_rgba())[..., :3].copy()


# ---------------------------------------------------------------- plots


def plot_reference_effect(mpc, theta_max, scenarios, values=(-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)):
    """Planned angle and force of the MPC for several actions a (theta_ref = theta_max a^3),
    starting from each state in scenarios = {name: (x, theta, v, omega)}."""
    names, values = list(scenarios), np.asarray(values)
    x0 = torch.tensor(list(scenarios.values()), dtype=torch.float32).repeat_interleave(len(values), 0)
    theta_ref = theta_max * torch.tensor(np.tile(values, len(names)), dtype=torch.float32)[:, None] ** 3
    with torch.no_grad():
        _, _, plan, plan_u, _ = mpc(x0, params={"xref1": theta_ref})
    plan = plan.reshape(len(names), len(values), *plan.shape[1:]).numpy()
    plan_u = plan_u.reshape(len(names), len(values), -1).numpy()
    t = np.arange(plan.shape[2]) * 0.05

    fig, axes = plt.subplots(2, len(names), figsize=(4.4 * len(names), 7), sharex=True, sharey="row")
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(values)))
    for j, name in enumerate(names):
        # Draw a = 0 last and dashed, so that lines below it stay visible.
        for i in sorted(range(len(values)), key=lambda i: values[i] == 0):
            style = dict(color="k", ls="--", lw=2) if values[i] == 0 else dict(color=colors[i], lw=2)
            axes[0, j].plot(t, plan[j, i, :, 1], label=f"$a={values[i]:g}$", **style)
            axes[1, j].step(t[:-1], plan_u[j, i], where="post", **style)
        axes[0, j].set_title(name)
        axes[1, j].set_xlabel("Time [s]")
        for ax in axes[:, j]:
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel(r"Planned $\theta$ [rad]")
    axes[1, 0].set_ylabel("Planned $F$ [N]")
    # Legend in the order of the values, with a = 0 in the middle.
    handles, labels = axes[0, -1].get_legend_handles_labels()
    order = np.argsort([float(label[3:-1]) for label in labels])
    axes[0, -1].legend([handles[k] for k in order], [labels[k] for k in order],
                       loc="center left", bbox_to_anchor=(1.02, 0.0))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- training and evaluation


class EpisodeReturns(BaseCallback):
    """Collects the return of every finished training episode (needs a VecMonitor).

    If given a marimo progress bar, it also advances the bar and shows the mean return
    of the last 20 episodes.
    """

    def __init__(self, progress=None):
        super().__init__()
        self.returns, self.progress = [], progress

    def _on_step(self):
        self.returns += [info["episode"]["r"] for info in self.locals["infos"] if "episode" in info]
        if self.progress is not None:
            recent = f"mean return of the last 20 episodes: {np.mean(self.returns[-20:]):.2f}" if self.returns else ""
            self.progress.update(increment=self.training_env.num_envs, subtitle=recent)
        return True


def plot_training(returns, window=20):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    returns = np.asarray(returns)
    ax.plot(returns, alpha=0.4)
    if len(returns) >= window:
        ax.plot(np.arange(window - 1, len(returns)), np.convolve(returns, np.ones(window) / window, "valid"))
    ax.set(xlabel="Training episode", ylabel="Return")
    fig.tight_layout()
    return fig


def rollout(make_env, policy, steps=80):
    """One episode from the hanging state: states (k + 1, 4), inputs (k, 2), i.e. the
    force and the angle reference, and the return."""
    env = make_env()
    obs, _ = env.reset()
    states, inputs, total = [env.unwrapped.s.copy()], [], 0.0
    for _ in range(steps):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += reward
        inputs.append([env.force[0], env.theta_ref[0]])
        states.append(env.unwrapped.s.copy())
        if terminated or truncated:
            break
    states = np.array(states)
    states[:, 1] = np.unwrap(states[:, 1])  # no jumps between -pi and pi ...
    states[:, 1] -= 2 * np.pi * np.round(states[-1, 1] / (2 * np.pi))  # ... and upright at the end is 0
    return states, np.array(inputs), total


def plot_closed_loop(results, dt=0.05, Fmax=80.0):
    """Closed loops {label: (states, inputs, return)} with the plotting function of Exercise 07."""
    plot_trajectories(
        x_traj_list=[states for states, _, _ in results.values()],
        u_traj_list=[inputs for _, inputs, _ in results.values()],
        time_traj_list=[np.arange(len(states)) * dt for states, _, _ in results.values()],
        labels_list=[f"{label} (return {total:.1f})" for label, (_, _, total) in results.items()],
        x_labels=["$x$ [m]", r"$\theta$ [rad]", "$v$ [m/s]", r"$\omega$ [rad/s]"],
        u_labels=["$F$ [N]", r"$\theta_\mathrm{ref}$ [rad]"],
        time_label="Time [s]",
        idxbu=[0],
        lbu=np.array([-Fmax]),
        ubu=np.array([Fmax]),
        ncol_legend=1,
        show_plot=False,
        latexify=False,
    )
    return plt.gcf()


def encode_gif(frames, duration=50):
    if not frames:
        return None
    images = [Image.fromarray(frame) for frame in frames]
    buffer = BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=duration, loop=0)
    return buffer.getvalue()


def rollout_gif(make_env, policy):
    """Animation of one episode, with the planned path of the pole tip in grey."""
    env = make_env(render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []
    for _ in range(80):
        state = env.unwrapped.s.copy()  # the plan of this step starts here
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        frames.append(env.unwrapped.draw(state, env.plan))
        if terminated or truncated:
            break
    return encode_gif(frames)
