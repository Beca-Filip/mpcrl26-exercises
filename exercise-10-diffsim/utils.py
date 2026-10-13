"""Helpers for Exercise 10 that are not part of the `wec` package: closed-loop evaluation of a
policy on the differentiable plant, the training configuration, the gradient check, result
table and plots.

A policy is anything with the call signature of a `ParameterizedController` without parameters,
    ctx, u = policy(obs, ctx=None)
for a batch of observations obs (B, 4) = [x, v, w1, w2] and PTO forces u (B, 1) in Newtons:
`LinearController`, or the notebook's `MpcPolicy` around the MPC controller.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Arc, ConnectionPatch

from wec import WecDynamics, polar

# --------------------------------------------------------------------------- closed loop
def simulate(plant: WecDynamics, policy, n_steps: int) -> dict[str, np.ndarray]:
    """Closed-loop rollout of one plant from rest at a wave crest, one solve per step and no warm
    start. Pe[i] is the mean electrical power over sample i [W]."""
    obs = plant.reset(torch.zeros(1, dtype=torch.float64))
    log = {k: np.zeros(n_steps) for k in ("t", "x", "v", "u", "Pe")}
    with torch.no_grad():
        for i in range(n_steps):
            _, u = policy(obs)
            log["t"][i], log["x"][i], log["v"][i], log["u"][i] = i * plant.dt, obs[0, 0], obs[0, 1], u[0, 0]
            obs, energy = plant.step(obs, u)
            log["Pe"][i] = float(energy[0]) / plant.dt
    return log


def steady_mean(sig: np.ndarray, n_per: int, n_periods: int = 5) -> float:
    """Mean over the last n_periods wave periods."""
    return float(np.mean(sig[-n_per * n_periods:]))


def evaluate(policy, plant: WecDynamics, periods: int = 20) -> dict:
    """Closed loop of a policy on the plant from rest; the power is the mean over the last 5
    periods."""
    log = simulate(plant, policy, periods * plant.n_per)
    return dict(power=steady_mean(log["Pe"], plant.n_per), xmax=float(np.abs(log["x"]).max()), log=log)


# --------------------------------------------------------------------------- RL configuration
REWARD_SCALE = 1e5   # [J] per unit of reward


@dataclass
class ShacConfig:
    x_max: float                 # stroke limit [m]: reward penalty and state scaling
    horizon: int = 10            # h: rollout length per iteration [steps]
    n_env: int = 16              # parallel plants = batch size of the MPC solver
    iters: int = 150             # actor updates
    gamma: float = 0.99
    lam: float = 0.95            # TD(lambda)
    actor_lr: float = 0.01       # Adam, decays linearly to 10 %
    critic_lr: float = 1e-3
    critic_epochs: int = 16
    critic_warmup: int = 50      # critic-only iterations before the actor moves
    target_alpha: float = 0.4    # target critic: phi' <- alpha phi' + (1 - alpha) phi
    episode_len: int = 80        # steps between random restarts
    restart_scale: float = 0.7   # restart buoy state: uniform in +-scale (x_max, omega x_max)
    grad_clip: float = 1.0
    viol_weight: float = 1e8     # stroke-violation penalty in the reward [J/m^2]
    seed: int = 0


def rollout(policy, plant: WecDynamics, obs0: torch.Tensor, steps: int = 20):
    """Energy of a closed-loop rollout from obs0 (B, 4), summed over plants and steps, in units of REWARD_SCALE."""
    obs, R = obs0, 0.0
    for _ in range(steps):
        _, u = policy(obs)
        obs, e = plant.step(obs, u)
        R = R + e.sum() / REWARD_SCALE
    return R


def test_rollout_gradient(policy, plant: WecDynamics) -> None:
    """Autograd dR/dg of a 20-step rollout on 4 plants (half near the stroke limit) vs central
    finite differences. Resets policy.g to the nominal (1, 0)."""
    rng = np.random.default_rng(3)
    z0, w0 = rng.normal(size=(4, 2)) * 0.8, rng.normal(size=(4, 2)) * 0.4
    z0[:2, 0] = rng.choice([-1, 1], size=2) * rng.uniform(1.3, 1.5, size=2)
    obs0 = torch.from_numpy(np.concatenate([z0, w0], axis=1))
    g0 = torch.tensor([0.9, 0.2], dtype=torch.float64)
    with torch.no_grad():
        policy.g.copy_(g0)
    grad = torch.autograd.grad(rollout(policy, plant, obs0), policy.g)[0]   # one backward pass through 20 solves and 20 plant steps
    fd = np.zeros(2)
    for i in range(2):
        vals = []
        for s in (1, -1):
            with torch.no_grad():
                policy.g.copy_(g0)
                policy.g[i] += s * 1e-5
            vals.append(rollout(policy, plant, obs0).item())
        fd[i] = (vals[0] - vals[1]) / 2e-5
    with torch.no_grad():
        policy.g.copy_(torch.tensor([1.0, 0.0]))   # back to the nominal model
    print(f"dR/dg autograd {grad.numpy()}, finite differences {fd}, "
          f"relative error {np.abs(grad.numpy() - fd).max() / np.abs(fd).max():.1e} (expected < 1e-3)")


def summarize(policy, plant: WecDynamics) -> dict:
    """Training callback: steady-state power and the policy's excitation coefficient."""
    gain, phase = polar(policy.g.detach().numpy())
    ev = evaluate(policy, plant)
    return dict(power=ev["power"], xmax=ev["xmax"], gain=gain, phase=phase)


def result_table(rows: list[tuple], P_nom: float, P_orc: float) -> str:
    """Markdown table. rows: (name, power [W], xmax [m], gain, phase [deg]); gain may be None."""
    out = ["| controller | P_e [kW] | of gap | max abs x [m] | gain | phase [deg] |",
           "|---|---|---|---|---|---|"]
    for name, P, xmax, gain, phase in rows:
        gp = f"{gain:.3f} | {phase:.1f}" if gain is not None else "- | -"
        out.append(f"| {name} | {P/1e3:.2f} | {100*(P - P_nom)/(P_orc - P_nom):.0f} % | {xmax:.3f} | {gp} |")
    return "\n".join(out)


# --------------------------------------------------------------------------- plots
C_PAS, C_MPC, C_NOM, C_SHAC = "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
_KEYS = (("SHAC", C_SHAC), ("nominal", C_NOM), ("passive", C_PAS), ("damper", C_PAS))


def _color(label: str) -> str:
    return next((c for key, c in _KEYS if key in label), C_MPC)


def plot_excitation(g_true, wt1: float = 60.0):
    """Excitation force of the nominal model g = (1, 0) and of the plant g_true, normalised by Gamma A:
    over one wave period (left) and as phasors at the instant wt1 [deg] (right). The phasors turn
    counter-clockwise from wt = 0, the wave crest, at the top; a phasor's height is the force."""
    gain, phase = polar(g_true)
    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True,
                                     gridspec_kw=dict(width_ratios=[1.8, 1.2]))
    wt = np.linspace(0, 360, 361)
    th = np.linspace(0, 2 * np.pi, 200)
    curves = (("nominal model g = (1, 0)", 1.0, 0.0, C_NOM),
              (f"plant g* = {gain:.1f} (cos {phase:.0f}°, sin {phase:.0f}°)", gain, phase, C_MPC))
    ax_t.axvline(wt1, color="0.6", lw=0.8)
    ax_t.text(wt1, -1.22, r" $\omega t_1$", color="0.4", ha="left", va="bottom")
    for label, a, lag, color in curves:
        ax_t.plot(wt, a * np.cos(np.radians(wt - lag)), color=color, lw=2, label=label)
        for y in (a, -a):   # amplitude lines across both panels
            ax_t.axhline(y, color=color, lw=0.6, ls="--")
            ax_p.axhline(y, color=color, lw=0.6, ls="--")
        ax_p.plot(a * np.cos(th), a * np.sin(th), color=color, lw=0.8, ls="--")
        ang = np.radians(wt1 - lag)
        tip = (-a * np.sin(ang), a * np.cos(ang))   # counter-clockwise from the top
        ax_p.annotate("", xy=tip, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=2, color=color))
        ax_t.plot(wt1, tip[1], "o", color=color)
        fig.add_artist(ConnectionPatch((wt1, tip[1]), tip, "data", "data", axesA=ax_t, axesB=ax_p,
                                       color=color, lw=1, ls=":"))
    # The lag in time: the downward zero crossings at 90 deg and 90 deg + phase.
    ax_t.annotate("", xy=(90 + phase, 0), xytext=(90, 0), arrowprops=dict(arrowstyle="<->", color="0.2"))
    ax_t.text(90 + phase / 2, 0.06, f"{phase:.0f}°", ha="center", va="bottom")
    ax_t.set(xlabel=r"wave phase $\omega t$ [deg] (0: crest)", ylabel=r"$F_{ex} / (\Gamma A)$",
             xticks=[0, 90, 180, 270, 360], xlim=(0, 360), ylim=(-1.25, 1.25),
             title="Excitation force over one wave period")
    ax_t.legend(fontsize=8, loc="lower right")
    # The lag as an angle: an arc between the two phasors, and the direction of rotation.
    ax_p.add_patch(Arc((0, 0), 0.7, 0.7, theta1=90 + wt1 - phase, theta2=90 + wt1, color="0.2"))
    mid = np.radians(90 + wt1 - phase / 2)
    ax_p.text(0.45 * np.cos(mid), 0.45 * np.sin(mid), f"{phase:.0f}°", ha="center", va="center")
    ax_p.annotate("", xy=(1.12 * np.cos(np.radians(-10)), 1.12 * np.sin(np.radians(-10))),
                  xytext=(1.12 * np.cos(np.radians(-50)), 1.12 * np.sin(np.radians(-50))),
                  arrowprops=dict(arrowstyle="-|>", color="0.2", connectionstyle="arc3,rad=0.2"))
    ax_p.text(1.05, -0.55, r"$\omega$", ha="left", va="center")
    for text, xy, ha, va in (("0° (crest)", (0, 1.04), "center", "bottom"), ("90°", (-1.04, 0), "right", "center"),
                             ("180°", (0, -1.04), "center", "top"), ("270°", (1.04, 0), "left", "center")):
        ax_p.text(*xy, text, ha=ha, va=va, fontsize=8, color="0.3")
    ax_p.plot([-1.05, 1.05], [0, 0], color="0.6", lw=0.6)
    ax_p.plot([0, 0], [-1.05, 1.05], color="0.6", lw=0.6)
    ax_p.set(xlim=(-1.45, 1.45), title=r"Phasors, turning at $\omega$")
    ax_p.set_aspect("equal", adjustable="box")
    ax_p.axis("off")
    ax_t.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_closed_loop(logs: dict[str, dict], x_max: float, u_max: float, n_per: int):
    """Heave, PTO force and electrical power over the last five periods (the window of the mean
    power)."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 9))
    first = next(iter(logs.values()))
    sl = slice(-5 * n_per, None)
    ts = first["t"][sl] - first["t"][sl][0]
    axs[0].axhspan(-x_max, x_max, color="0.9", lw=0, label=f"stroke limit +-{x_max:g} m")
    for label, lg in logs.items():
        axs[0].plot(ts, lg["x"][sl], color=_color(label), lw=1.5, label=label)
    axs[0].set(xlabel="time in the last five periods [s]", ylabel="x [m]", title="Heave")
    for lim in (u_max, -u_max):
        axs[1].axhline(lim / 1e3, color="0.5", lw=0.8)
    for label, lg in logs.items():
        axs[1].step(ts, lg["u"][sl] / 1e3, where="post", color=_color(label), lw=1.5, label=label)
    axs[1].set(xlabel="time in the last five periods [s]", ylabel="u [kN]", title="PTO force")
    axs[2].axhline(0, color="0.5", lw=0.8)
    for label, lg in logs.items():
        axs[2].step(ts, lg["Pe"][sl] / 1e3, where="post", color=_color(label), lw=1.5,
                    label=f"{label}: mean {steady_mean(lg['Pe'], n_per) / 1e3:.1f} kW")
    axs[2].set(xlabel="time in the last five periods [s]", ylabel="P_e [kW]",
               title="Electrical power (negative = motoring)")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    return fig


def plot_training(log: list[dict], P_nom: float, P_orc: float, g_true):
    """Left: steady-state power during training. Right: the path of g from the nominal model
    towards the true coefficient. log entries carry env_steps, power, gain, phase."""
    ev = [e for e in log if "power" in e]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axs[0]
    for P, text in ((P_orc, "MPC with the true excitation model"), (P_nom, "nominal model (start)")):
        ax.axhline(P / 1e3, color="0.5", lw=0.8)
        ax.annotate(text, (1, P / 1e3), xycoords=("axes fraction", "data"), fontsize=8,
                    ha="right", va="bottom", color="0.3")
    ax.plot([e["env_steps"] / 1e3 for e in ev], [e["power"] / 1e3 for e in ev], "-o",
            color=C_SHAC, lw=2, ms=3, label="SHAC")
    ax.set(xlabel="simulated environment steps [thousands]", ylabel="P_e [kW]",
           title="Steady-state power of the MPC during training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    ax = axs[1]
    ax.plot([e["phase"] for e in ev], [e["gain"] for e in ev], "-o", color=C_SHAC, lw=2, ms=3, label="SHAC")
    ax.plot(0, 1, "o", color=C_NOM, ms=10, mec="w", mew=1.5, label="nominal model (start)")
    gain, phase = polar(g_true)
    ax.plot(phase, gain, "*", color=C_MPC, ms=16, mec="w", mew=1.2, label="true coefficient")
    ax.set(xlabel="phase lag of the excitation force [deg]", ylabel="gain [-]", ylim=(0, 1.2),
           title="Path of the MPC's excitation coefficient g")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig
