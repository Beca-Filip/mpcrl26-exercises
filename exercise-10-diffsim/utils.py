"""Helpers for Exercise 10 that are not part of the `wec` package: closed-loop evaluation of a
policy on the Gymnasium environment, the power landscape, the training configuration, the
gradient check, result table and plots.

A policy is anything with the call signature of a `ParameterizedController` without parameters,
    ctx, u = policy(obs, ctx=None)
for a batch of observations obs (B, 4) = [x, v, w1, w2] and PTO forces u (B, 1) in Newtons:
`LinearController`, or the notebook's `FixedMpcPolicy` and `MpcPolicy` around the MPC controller.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from wec import WecDynamics, WecEnv, WecPlanner, g_polar, polar

# --------------------------------------------------------------------------- closed loop
def simulate(env: WecEnv, policy, n_steps: int) -> dict[str, np.ndarray]:
    """Closed-loop rollout from rest, one solve per step and no warm start. Pe[i] is the mean
    electrical power over sample i [W]."""
    obs, _ = env.reset()
    log = {k: np.zeros(n_steps) for k in ("t", "x", "v", "u", "Fex", "Pe")}
    for i in range(n_steps):
        with torch.no_grad():
            _, a = policy(torch.from_numpy(obs)[None])
        u = float(a[0, 0])
        log["x"][i], log["v"][i], log["u"][i] = obs[0], obs[1], u
        obs, _, _, _, info = env.step(np.array([u]))
        log["t"][i], log["Fex"][i], log["Pe"][i] = info["t"], info["Fex"], info["Pe"]
    return log


def steady_mean(sig: np.ndarray, n_per: int, n_periods: int = 5) -> float:
    """Mean over the last n_periods wave periods."""
    return float(np.mean(sig[-n_per * n_periods:]))


def fundamental(sig: np.ndarray, n_per: int) -> complex:
    """Complex amplitude of the fundamental of sig over its last period, s(t) = Re{S e^{j omega t}}."""
    e = np.exp(-2j * np.pi * np.arange(n_per) / n_per)
    return complex(2 * (sig[-n_per:] @ e) / n_per)


def phase_deg(ref: np.ndarray, sig: np.ndarray, n_per: int) -> float:
    """Phase of sig relative to ref over the last period [deg]; positive means sig leads."""
    return float(np.degrees(np.angle(fundamental(sig, n_per) / fundamental(ref, n_per))))


def evaluate(policy, env: WecEnv, periods: int = 20) -> dict:
    """Closed loop of a policy on the environment from rest; steady-state numbers are means over
    the last 5 periods."""
    log = simulate(env, policy, periods * env.n_per)
    return dict(power=steady_mean(log["Pe"], env.n_per), xmax=float(np.abs(log["x"]).max()),
                umax=float(np.abs(log["u"]).max()),
                v_phase=phase_deg(log["Fex"], log["v"], env.n_per), log=log)


def power_landscape(planner: WecPlanner, env: WecEnv, gains, phases, periods: int = 12) -> dict:
    """Closed-loop mean power of the MPC on the plant env for every (gain, phase) of its
    excitation coefficient. All grid points run as one batch of plants (about 80 000 solves)."""
    G = np.array([g_polar(gn, ph) for gn in gains for ph in phases])          # (B, 2)
    Bsz, n_steps = len(G), periods * env.n_per
    g_t = torch.from_numpy(G)                                                   # differentiable parameter: (B, 2) tensor
    obs = env.dynamics.reset(torch.zeros(Bsz, dtype=torch.float64))            # every plant from rest at a crest
    Pe = np.zeros((Bsz, n_steps))
    with torch.no_grad():
        for i in range(n_steps):
            u = planner(obs, params={"g": g_t})[1]
            obs, energy = env.dynamics.step(obs, u)
            Pe[:, i] = energy.numpy() / env.d.dt
    P = Pe[:, -5 * env.n_per:].mean(axis=1).reshape(len(gains), len(phases))
    return dict(gain=np.asarray(gains), phase=np.asarray(phases), P=P)


# --------------------------------------------------------------------------- RL configuration
REWARD_SCALE = 1e5   # [J] per unit of reward


@dataclass
class ShacConfig:
    x_max: float                 # stroke limit [m]: reward penalty and state scaling
    horizon: int = 10            # h: rollout length per iteration [steps]
    n_env: int = 16              # parallel plants = batch size of the MPC solver
    iters: int = 150             # actor updates
    gamma: float = 0.99
    critic: bool = True          # False: truncated BPTT (no terminal value)
    lam: float = 0.95            # TD(lambda)
    actor_lr: float = 0.02       # Adam, decays linearly to 10 %
    critic_lr: float = 1e-3
    critic_epochs: int = 16
    critic_warmup: int = 50      # critic-only iterations before the actor moves
    target_alpha: float = 0.4    # target critic: phi' <- alpha phi' + (1 - alpha) phi
    episode_len: int = 80        # steps between random restarts
    restart_scale: float = 0.7   # restart buoy state: uniform in +-scale (x_max, omega x_max)
    action_noise: float = 0.0    # Gaussian noise on the PTO force while training [N]
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


def summarize(policy, env: WecEnv) -> dict:
    """Training callback: steady-state power and the policy's excitation coefficient."""
    gain, phase = polar(policy.g.detach().numpy())
    ev = evaluate(policy, env)
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
C_LQ, C_PAS = "#2a78d6", "#eb6834"
C_MPC, C_NOM, C_SHAC, C_BPTT, C_SHORT = "#1baf7a", "#eda100", "#4a3aa7", "#008300", "#e87ba4"
_KEYS = (("truncated", C_SHORT), ("SHAC", C_SHAC), ("BPTT", C_BPTT), ("nominal", C_NOM),
         ("LQ", C_LQ), ("passive", C_PAS), ("damper", C_PAS))


def _color(label: str) -> str:
    return next((c for key, c in _KEYS if key in label), C_MPC)


def plot_closed_loop(logs: dict[str, dict], x_max: float, u_max: float, n_per: int):
    """Heave over the whole run, PTO force and electrical power over the last two periods."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 9))
    first = next(iter(logs.values()))
    sl = slice(-2 * n_per, None)
    ts = first["t"][sl] - first["t"][sl][0]
    axs[0].axhspan(-x_max, x_max, color="0.9", lw=0, label=f"stroke limit +-{x_max:g} m")
    for label, lg in logs.items():
        axs[0].plot(lg["t"], lg["x"], color=_color(label), lw=1.5, label=label)
    axs[0].set(xlabel="time [s]", ylabel="x [m]", title="Heave")
    for lim in (u_max, -u_max):
        axs[1].axhline(lim / 1e3, color="0.5", lw=0.8)
    for label, lg in logs.items():
        axs[1].step(ts, lg["u"][sl] / 1e3, where="post", color=_color(label), lw=1.5, label=label)
    axs[1].set(xlabel="time in the last two periods [s]", ylabel="u [kN]", title="PTO force")
    axs[2].axhline(0, color="0.5", lw=0.8)
    for label, lg in logs.items():
        axs[2].step(ts, lg["Pe"][sl] / 1e3, where="post", color=_color(label), lw=1.5,
                    label=f"{label}: mean {steady_mean(lg['Pe'], n_per) / 1e3:.1f} kW")
    axs[2].set(xlabel="time in the last two periods [s]", ylabel="P_e [kW]",
               title="Electrical power (negative = motoring)")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    return fig


def plot_phasors(logs: dict[str, dict], n_per: int):
    """Velocity against the plant's excitation force: traces (left) and phasors (right).

    Reactive control wants v in phase with F_ex. The right panel draws F_ex as the reference
    arrow at angle 0 and one velocity arrow per controller at its relative phase.
    """
    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    first = next(iter(logs.values()))
    sl = slice(-2 * n_per, None)
    ts = first["t"][sl] - first["t"][sl][0]
    ax_t.plot(ts, first["Fex"][sl] / np.abs(first["Fex"][-n_per:]).max(), "k--", lw=1.2, label="F_ex (plant)")
    for label, lg in logs.items():
        ax_t.plot(ts, lg["v"][sl] / np.abs(lg["v"][-n_per:]).max(), color=_color(label), lw=1.5, label=f"v, {label}")
    ax_t.set(xlabel="time in the last two periods [s]", ylabel="normalised [-]", ylim=(-1.6, 1.2),
             title="Velocity vs excitation force")
    ax_t.grid(alpha=0.3)
    ax_t.legend(fontsize=8, loc="lower right", ncol=3)

    # Unit-length arrows at the phase of v relative to the plant's F_ex (amplitudes not shown).
    ref = fundamental(first["Fex"], n_per)
    arrows = {label: np.exp(1j * np.angle(fundamental(lg["v"], n_per) / ref)) for label, lg in logs.items()}
    ax_p.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2, color="k"))
    ax_p.text(1.02, 0.0, "F_ex", ha="left", va="center")
    for label, ph in arrows.items():
        ax_p.annotate("", xy=(ph.real, ph.imag), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="->", lw=2, color=_color(label)))
        ax_p.text(1.08 * ph.real, 1.08 * ph.imag, f"{label}: {np.degrees(np.angle(ph)):+.0f} deg",
                  color=_color(label), ha="left" if ph.real >= 0 else "right", va="center", fontsize=8)
    th = np.linspace(0, 2 * np.pi, 200)
    ax_p.plot(np.cos(th), np.sin(th), color="0.8", lw=0.8)
    ax_p.set(xlim=(-1.3, 1.6), ylim=(-1.3, 1.3), aspect="equal", title="Phasors (counter-clockwise = leads)")
    ax_p.axis("off")
    fig.tight_layout()
    return fig


def plot_landscape(land: dict, g_true, paths: dict | None = None, ax=None):
    """Contours of the closed-loop power over (phase, gain) of the MPC's excitation coefficient,
    the nominal start, the true coefficient and optional training paths."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.8))
    if land is not None:
        cs = ax.contour(land["phase"], land["gain"], land["P"] / 1e3,
                        levels=[20, 25, 28, 30, 32, 33, 33.4], colors="0.4", linewidths=0.8)
        ax.clabel(cs, fmt="%g kW", fontsize=7)
        ax.set(xlim=(land["phase"][0], land["phase"][-1]), ylim=(0, land["gain"][-1]))
    for r in (paths or {}).values():
        ev = [e for e in r["log"] if "power" in e]
        ax.plot([e["phase"] for e in ev], [e["gain"] for e in ev], "-o", color=r["color"],
                lw=2, ms=3, label=r["label"])
    ax.plot(0, 1, "o", color=C_NOM, ms=10, mec="w", mew=1.5, label="nominal model (start)")
    gain, phase = polar(g_true)
    ax.plot(phase, gain, "*", color=C_MPC, ms=16, mec="w", mew=1.2, label="true coefficient")
    ax.set(xlabel="phase lag of the excitation force [deg]", ylabel="gain [-]",
           title="Closed-loop power landscape P_e(gain, phase)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    return ax.figure


def plot_training(runs: dict, P_nom: float, P_orc: float, g_true, land: dict | None):
    """Left: steady-state power during training. Right: the paths of g on the power landscape.
    runs: name -> dict(label, color, log) with log entries carrying env_steps, power, gain, phase."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axs[0]
    for P, text in ((P_orc, "MPC with the true excitation model"), (P_nom, "nominal model (start)")):
        ax.axhline(P / 1e3, color="0.5", lw=0.8)
        ax.annotate(text, (1, P / 1e3), xycoords=("axes fraction", "data"), fontsize=8,
                    ha="right", va="bottom", color="0.3")
    for r in runs.values():
        ev = [e for e in r["log"] if "power" in e]
        ax.plot([e["env_steps"] / 1e3 for e in ev], [e["power"] / 1e3 for e in ev], "-o",
                color=r["color"], lw=2, ms=3, label=r["label"])
    ax.set(xlabel="simulated environment steps [thousands]", ylabel="P_e [kW]",
           title="Steady-state power of the MPC during training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    plot_landscape(land, g_true, paths=runs, ax=axs[1])
    fig.tight_layout()
    return fig
