# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.25.1",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import time
    from copy import deepcopy

    import marimo as mo
    import numpy as np
    import torch
    from torch import nn
    from utils import (
        REWARD_SCALE,
        ShacConfig,
        evaluate,
        plot_closed_loop,
        plot_excitation,
        plot_training,
        result_table,
        summarize,
        test_rollout_gradient,
    )
    from wec import (
        ControllerFromPlanner,
        LinearController,
        WECParams,
        WecDynamics,
        WecPlanner,
        WecPlannerConfig,
        discretise,
        g_polar,
        optimal_damper,
        polar,
    )

    return (
        ControllerFromPlanner,
        LinearController,
        REWARD_SCALE,
        ShacConfig,
        WECParams,
        WecDynamics,
        WecPlanner,
        WecPlannerConfig,
        deepcopy,
        discretise,
        evaluate,
        g_polar,
        mo,
        nn,
        np,
        optimal_damper,
        plot_closed_loop,
        plot_excitation,
        plot_training,
        polar,
        result_table,
        summarize,
        test_rollout_gradient,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 10 — Learning an MPC through a differentiable simulator (Solution)

    **Goal:** a wave energy converter is controlled by an economic MPC whose model of the wave
    force is wrong in a single coefficient $g$. You will (1) see where this mismatch comes from,
    (2) measure how much power the MPC with the nominal $g$ loses against the same MPC with the
    true $g$, and (3) recover most of it by training $g$ with short-horizon actor-critic (SHAC)
    through a differentiable simulation of plant and MPC.

    ## 1. A wave energy converter and its excitation coefficient

    A heaving point absorber: a hemispherical buoy (radius 2.5 m, about 59 t including added
    mass) in a regular wave of height 1 m and period 8 s. The power take-off (PTO) applies a
    force $u$ on the buoy. In the linear potential-flow model the heave $x$ obeys

    $$
    M\ddot x = -k x - B \dot x + F_{ex}(t) + u ,
    $$

    with hydrostatic stiffness $k$, radiation damping $B$ and the wave excitation force $F_{ex}$.
    The regular wave comes from a two-state exosystem, a rotation at the wave frequency $\omega$:

    $$
    w = A\begin{bmatrix}\cos\omega t\\ \sin\omega t\end{bmatrix},\qquad
    \dot w = \begin{bmatrix}0 & -\omega\\ \omega & 0\end{bmatrix} w,\qquad
    F_{ex} = \Gamma\,(g_1 w_1 + g_2 w_2).
    $$

    The **excitation coefficient** $g = (g_1, g_2)$ scales the force by $|g|$ and delays it by
    the angle of $g$. The long-wave model, and with it the MPC below, has $g = (1, 0)$. The plant
    in this exercise has $g^\star = 0.8\,(\cos 40^\circ, \sin 40^\circ)$: its excitation force
    is 20 % weaker and lags the wave by 40 degrees. The figure shows both forces over one wave
    period and as phasors, arrows that turn counter-clockwise at $\omega$ and whose height is
    the force.
    """)
    return


@app.cell
def _(G_TRUE, plot_excitation):
    plot_excitation(G_TRUE)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Where the mismatch comes from.** The excitation force cannot be measured. It is
    reconstructed from a wave estimate (a sensor on the buoy, a wave rider some distance
    up-wave, a forecast), its amplitude and phase relative to the wave elevation depend on the
    diffraction correction of the hydrodynamic model, and the buoy itself moves. At one wave
    frequency all of these errors are a single complex number, a gain and a phase: $g$.

    **Power and constraints.** The PTO absorbs the mechanical power $-u\dot x$ and loses
    $r u^2$ in its windings, so the electrical power is $P_e = -u v - r u^2$ and the controller
    minimises the stage cost $\ell = u v + r u^2 = -P_e$, an economic objective. The stroke is
    limited to $|x| \le 1.5$ m (soft: end stops) and the force to $|u| \le 300$ kN (hard).

    **Discrete time and the simulator.** With a zero-order hold on $u$ and $\Delta t = 0.2$ s
    the sampled-data model is exact: $z^+ = A z + G_w w + B_u u$ and $w^+ = \Phi_w w$ with
    $z = (x, v)$, and the electrical energy of one sample is $e = -y^\top W_d\, y$ with
    $y = (z, w, u)$ (`discretise` in `wec/dynamics.py`, Van Loan integral). `WecDynamics` is this
    plant written as pure tensor functions: `reset(phase)` returns buoys at rest in waves of the
    given phases, and `step(obs, u)` maps $\mathrm{obs} = (z, w)$ and the force to
    $(\mathrm{obs}^+, e)$. It stores no state, and the leading dimension is a free batch of
    plants. It is the only simulator in this notebook: `evaluate` steps one plant through it
    under `torch.no_grad()`, and the training loop of Section 4 steps a batch of plants with
    autograd recording the whole rollout.
    """)
    return


@app.cell
def _(WECParams, WecDynamics, discretise, g_polar, torch):
    DT, N, X_MAX, U_MAX = (
        0.2,
        20,
        1.5,
        300e3,
    )  # sample time [s], MPC horizon, stroke [m], force [N]
    G_TRUE = g_polar(
        0.8, 40.0
    )  # the plant's excitation coefficient; the MPC's model has (1, 0)
    p = WECParams()
    d_true = discretise(p, DT).with_excitation(G_TRUE)  # the plant's matrices
    plant = WecDynamics(d_true, p.amp)  # the true plant, batched and differentiable
    print(
        f"Inertia M = {p.M / 1e3:.1f} t, stiffness k = {p.k / 1e3:.0f} kN/m, radiation damping B = {p.B / 1e3:.1f} kNs/m"
    )
    print(
        f"Natural period {p.natural_period:.2f} s vs wave period {p.wave_period:.0f} s; "
        f"excitation force amplitude {p.F_amp / 1e3:.0f} kN; r = {p.r:.2e} 1/(N s/m); "
        f"{plant.n_per} samples per wave period"
    )
    _obs = plant.reset(
        torch.zeros(3, dtype=torch.float64)
    )  # three buoys at rest at a crest
    _u = torch.tensor([[0.0], [1e5], [-1e5]], dtype=torch.float64)
    _obs_next, _e = plant.step(_obs, _u)
    print(
        f"step: obs {tuple(_obs.shape)} and u {tuple(_u.shape)} -> obs' {tuple(_obs_next.shape)}, energy {tuple(_e.shape)}"
    )
    print(
        f"energy of the first sample from rest for u = 0, +100 kN, -100 kN: {_e.numpy().round(1)} J "
        "(pushing a resting buoy is motoring: negative)"
    )
    return DT, G_TRUE, N, U_MAX, X_MAX, d_true, p, plant


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Baseline: a passive damper

    The simplest PTO control is a linear damper $u = -c\,v$, which can only absorb power.
    `optimal_damper` picks the best $c$ for the plant (from the frequency domain, refined for the
    sampled loop). It is the reference the MPC is compared with, not the focus.

    `LinearController` in `wec/controller.py` is a `ParameterizedController` of leap-c-lab:
    `forward(obs, params=None, ctx=None)` maps a batch of observations to $u = -K\,\mathrm{obs}$
    and returns `(ctx, u)`; the damper is $K = (0, c, 0, 0)$. `evaluate` in `utils.py` runs any
    such policy on one plant for 20 wave periods from rest and averages the power over the last
    five.
    """)
    return


@app.cell
def _(LinearController, d_true, evaluate, optimal_damper, p, plant):
    c_pas = optimal_damper(d_true, p)
    ev_pas = evaluate(LinearController.damper(c_pas), plant)
    print(
        f"Damper c = {c_pas / 1e3:.1f} kNs/m: P_e {ev_pas['power'] / 1e3:.2f} kW, max|x| {ev_pas['xmax']:.3f} m "
        "(expected 6.44 kW, 0.335 m)"
    )
    return (ev_pas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Economic MPC and the cost of the wrong $g$

    The MPC uses the exact discrete model and cost, a wave forecast $w_k = \Phi_w^k w_0$ over
    the horizon $N = 20$ (4 s), the two constraints, and the value function of the
    *unconstrained* average-cost LQ problem as terminal cost:

    $$
    \begin{aligned}
    \min_{u_{0:N-1},\,s}\quad & \sum_{k=0}^{N-1} y_k^\top W_d\, y_k
      \;+\; z_N^\top P z_N + 2\, z_N^\top Y w_N
      \;+\; \sum_{k=1}^{N} \big(\rho_1 s_k + \rho_2 s_k^2\big) \\
    \text{s.t.}\quad & z_{k+1} = A z_k + G_w \tilde w_k + B_u u_k,\qquad
      \tilde w_k = G(g)\, w_k,\qquad w_k = \Phi_w^k w_0, \\
    & |u_k| \le u_{\max},\qquad |x_k| \le x_{\max} + s_k,\qquad s_k \ge 0 .
    \end{aligned}
    $$

    $P$ solves a discrete algebraic Riccati equation and $Y$ a discrete Lyapunov equation (`design_lq`).
    The stroke limit is soft because a hard bound is not recursively feasible against a
    persistent wave. The OCP is given (`export_parametric_ocp` in `wec/acados_ocp.py`): acados
    with discrete dynamics, external cost and HPIPM, inputs scaled by $10^5$ N and costs by
    $10^3$ J.

    **Parameters in leap-c.** The OCP has two parameters. Both are registered with an
    `AcadosParameterManager`, which returns the CasADi symbols the OCP expressions are built
    from and later turns the values passed at solve time into acados parameters:

    | name | registered as | enters acados as | value passed to `params` |
    |---|---|---|---|
    | `g` | `differentiable=True` | `p_global`, sensitivities available | tensor of shape `(B, 2)` |
    | `wave` | default, non-differentiable | stage-wise `p` | numpy array of shape `(B, N+1, 2)`, built by the planner |

    `g` is the parameter we will learn. A differentiable MPC layer needs at least one
    differentiable parameter anyway: acados computes parametric sensitivities only for
    `p_global`.

    **The planner.** `WecPlanner` in `wec/planner.py` is a `ParameterizedPlanner` of leap-c-lab
    around leap-c's `AcadosDiffMpcTorch`, which generates and compiles the solver for a batch of
    `n_batch_init` problems that are solved in parallel. Its
    `forward(obs, action=None, params=None, ctx=None)` takes the observation $(z, w)$ of a batch
    of plants, uses $z$ as the initial state of the OCP, builds the forecast from $w$, and
    returns `(ctx, u0, x, u, value)`: the leap-c context (solver status, iterate for warm
    starts), the first input, the predicted trajectories and the cost, with forces in Newtons.
    `ControllerFromPlanner` turns the planner into a `ParameterizedController` that returns only
    `(ctx, u0)`: the MPC as a controller.
    """)
    return


@app.cell
def _(
    ControllerFromPlanner,
    DT,
    N,
    U_MAX,
    WecPlanner,
    WecPlannerConfig,
    X_MAX,
    p,
):
    planner = WecPlanner(
        WecPlannerConfig(
            N_horizon=N, dt=DT, x_max=X_MAX, u_max=U_MAX, params=p, n_batch_init=16
        )
    )  # a batch of 16: the plants trained in parallel in Section 4
    mpc = ControllerFromPlanner(
        planner
    )  # (obs, params, ctx) -> (ctx, u0): the MPC as a controller
    print(planner.param_manager)
    print(f"param_space {planner.param_space}, default_param {planner.default_param()}")
    return (mpc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The MPC as a torch module

    `MpcPolicy` wraps the MPC controller as an `nn.Module` whose only trainable parameter is the
    excitation coefficient `g` of its model. The same class gives the nominal MPC
    ($g = (1, 0)$), the oracle that knows $g^\star$, and the policy trained in Section 4. All of
    them share the one compiled solver, because `g` is a runtime parameter. Three details
    matter:

    * The planner sees `g` through the `params` dict. Pass `self.g` itself, expanded to the
      batch, not a copy: autograd must see the path from the parameter into the solver.
    * The forecast `wave` is a non-differentiable parameter. `WecPlanner.forward` builds it from
      the wave state with `obs[:, 2:].detach()`: the wave is exogenous, nothing the policy does
      changes it, and leap-c raises a `ValueError` if a tensor that requires grad is passed for
      a non-differentiable parameter. The buoy state `obs[:, :2]` keeps its graph.
    * Do not pass leap-c's `discount_factor`: it rescales the stage costs by $\gamma^k$ and
      would silently change this economic OCP.

    Complete `forward`.
    """)
    return


@app.cell
def _(nn, np, torch):
    class MpcPolicy(nn.Module):
        """u = MPC_g(obs) [N] for a batch of plants, differentiable in g and in the buoy state."""

        def __init__(self, controller, g_init=(1.0, 0.0)):
            super().__init__()
            self.controller = controller  # (obs, params, ctx) -> (ctx, u0)
            # The excitation coefficient of the MPC's model, learnable.
            self.g = nn.Parameter(torch.as_tensor(np.asarray(g_init, dtype=float)))

        def forward(self, obs, ctx=None):
            """obs: (B, 4). Returns the leap-c context (solver status, warm start) and the PTO force (B, 1) [N]."""
            # The parameter itself, expanded to the batch, graph intact.
            params = {"g": self.g.expand(obs.shape[0], 2)}
            return self.controller(obs, params=params, ctx=ctx)

    return (MpcPolicy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The nominal MPC on the true plant

    The MPC believes $g = (1, 0)$, the plant has $g^\star$. For reference, the **oracle** is the
    same MPC with $g^\star$ in its model: the best this MPC can do on this plant. Every
    controller runs on the same plant for 20 periods from rest.
    """)
    return


@app.cell
def _(G_TRUE, MpcPolicy, ev_pas, evaluate, mpc, plant):
    ev = {
        "MPC, nominal model": evaluate(MpcPolicy(mpc), plant),  # g = (1, 0)
        "MPC, true model (oracle)": evaluate(MpcPolicy(mpc, G_TRUE), plant),
        "passive damper": ev_pas,
    }
    for _name, _e in ev.items():
        print(f"{_name:26s} P_e {_e['power'] / 1e3:6.2f} kW  max|x| {_e['xmax']:.3f} m")
    print(
        "(expected: nominal 25.26 kW, 1.519 m; oracle 33.56 kW, 1.500 m; damper 6.44 kW, 0.335 m)"
    )
    return (ev,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Learning $g$ through the differentiable simulator

    The nominal MPC loses a quarter of the oracle's power. $g$ cannot be measured, but what
    counts can: the energy the plant delivers under the controller. So treat $g$ as a **policy
    parameter** and improve it by gradient ascent on the simulated closed-loop energy. The plant
    `WecDynamics` is already a differentiable torch function. The MPC must be differentiable in
    its parameter $g$ **and** in its initial state $z_0$, because the gradient flows through the
    state from one step to the next. leap-c provides $\partial u_0/\partial g$ and
    $\partial u_0/\partial z_0$ from the implicit function theorem on the KKT conditions of the
    converged QP, reusing HPIPM's factorisation (Frey et al. 2026).

    **Is it really differentiable?** Roll the policy out for 20 steps on 4 plants, sum the
    energy, and compare the gradient with respect to $g$ from autograd with central finite
    differences. Half of the initial states sit near the stroke limit so that constraints are
    active along the way.
    """)
    return


@app.cell
def _(MpcPolicy, mpc, plant, test_rollout_gradient):
    test_rollout_gradient(MpcPolicy(mpc), plant)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Short-horizon actor-critic (SHAC)

    **Objective.** Maximise the discounted closed-loop return of $n$ plants started from random
    wave phases and random buoy states,

    $$
    J(g) = \mathbb{E}\Big[\sum_{t \ge 0} \gamma^t r_t\Big],\qquad
    r_t = \frac{1}{10^5\,\mathrm{J}}\Big(e_t - \rho\,\operatorname{relu}\big(|x_{t+1}| - x_{\max}\big)^2\Big),
    $$

    where $e_t$ is the electrical energy of step $t$ from `plant.step` and the penalty with
    $\rho = 10^8$ J/m² keeps the learned controller inside the stroke that the MPC's soft
    constraint promises.

    **Loss.** Simulator and policy are differentiable, so the gradient of a window of $h$ steps
    is one `backward()` through $h$ plant steps and $h$ MPC solves: no action noise, no
    likelihood ratio. SHAC (Xu et al. 2022) closes the window with the value of the state it
    leaves behind:

    $$
    L(g) = -\frac{1}{h}\,\frac{1}{n}\sum_{i=1}^{n}\Big[\sum_{t=0}^{h-1}\gamma^t r_t^{(i)}
      + \gamma^h\Big(\Phi\big(s_h^{(i)}\big) + V_\phi\big(s_h^{(i)}\big)\Big)\Big],
    \qquad \Phi(s) = \frac{\tfrac12 k x^2 + \tfrac12 M v^2}{10^5\,\mathrm{J}} .
    $$

    The plants keep running across iterations, but the graph is cut every $h$ steps with
    `detach`. Without the value term the window would ignore the energy stored in the moving
    buoy (about 200 kJ, against a net yield of about 50 kJ per step) and reward *not* storing
    it, that is, a smaller excitation gain. That stored energy $\Phi$ is known, so only the
    rest of the value is learned, by a critic $V_\phi$: a small MLP on the scaled state
    $s = (z, w)$. This matters because the actor sees the value only through its slope at
    $s_h$, which a critic fitted to values gets only roughly. The critic is fitted by regression to
    TD($\lambda$) returns of the same rollouts with the shaped rewards
    $r_t' = r_t + \gamma\,\Phi(s_{t+1}) - \Phi(s_t)$,

    $$
    G_t = r_t' + \gamma\big[(1 - \lambda)\, V_{\bar\phi}(s_{t+1}) + \lambda\, G_{t+1}\big],
    \qquad G_h = V_{\bar\phi}(s_h),
    $$

    with a slowly moving target network $\bar\phi$. The $\Phi$ terms telescope, so
    $\Phi(s_t) + G_t$ is the TD($\lambda$) return of the full value. An untrained critic would
    bias the gradient, so the first `critic_warmup` iterations fit the critic only.

    **Batching and exploration.** The $n$ plants are one batch of the acados solver: one call
    `policy(obs)` solves $n$ OCPs in parallel. There is no action noise; diversity comes from
    the random wave phases and buoy states at the restarts every `episode_len` steps.

    `ShacConfig` in `utils.py` collects the settings. Complete `reward`, `td_lambda_targets` and
    the two marked lines of `train`, then read the loop: it is the whole pipeline.
    """)
    return


@app.cell
def _(REWARD_SCALE, p, torch):
    def reward(energy, x_next, cfg):
        """Electrical energy of the step minus the stroke-violation penalty, in units of REWARD_SCALE. All (B,)."""
        return (
            energy - cfg.viol_weight * torch.relu(x_next.abs() - cfg.x_max) ** 2
        ) / REWARD_SCALE

    def stored_energy(obs):
        """Phi: the energy stored in the moving buoy, 1/2 k x^2 + 1/2 M v^2, in units of REWARD_SCALE. obs: (..., 4)."""
        return (
            0.5 * p.k * obs[..., 0] ** 2 + 0.5 * p.M * obs[..., 1] ** 2
        ) / REWARD_SCALE

    return reward, stored_energy


@app.cell
def _(ShacConfig, reward, torch):
    print(
        "Reward for 100 kJ at x = 1.0 m and at x = 1.6 m (expected [1.0, -9.0]):",
        reward(
            torch.tensor([1e5, 1e5]), torch.tensor([1.0, 1.6]), ShacConfig(x_max=1.5)
        ),
    )
    return


@app.cell
def _(torch):
    def td_lambda_targets(R, v_next, gamma, lam):
        """TD(lambda) returns G_t for t = 0..h-1. R: rewards (h, B); v_next: target values V(s_{t+1}), (h, B)."""
        G, targets = v_next[-1], []
        for t in reversed(range(len(R))):
            # Mix the one-step bootstrap and the lambda-return.
            G = R[t] + gamma * ((1 - lam) * v_next[t] + lam * G)
            targets.append(G)
        return torch.stack(targets[::-1])

    return (td_lambda_targets,)


@app.cell
def _(td_lambda_targets, torch):
    print(
        "TD(lambda) targets (expected [[2.125], [2.5]]):",
        td_lambda_targets(
            torch.tensor([[1.0], [1.0]]),
            torch.tensor([[2.0], [3.0]]),
            gamma=0.5,
            lam=0.5,
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **TODO**: In the following, the SHAC training is implemented. Read the training loop carefully and try to build an understanding how the algorithm backpropagates from the loss to the learnable excitation parameters. You do not need to complete the code to make it run. Try playing around with different parameters and observe what happens.
    """)
    return


@app.cell
def _(deepcopy, nn, np, reward, stored_energy, td_lambda_targets, torch):
    def train(policy, plant, cfg, callback=None):
        """SHAC. Trains policy.parameters() in place; returns a log per iteration."""
        torch.manual_seed(cfg.seed)
        omega = float(torch.atan2(plant.Phiw[1, 0], plant.Phiw[0, 0])) / plant.dt
        scale = torch.tensor(
            [cfg.x_max, omega * cfg.x_max, plant.amp, plant.amp]
        )  # critic input scaling of obs = (z, w)
        critic = nn.Sequential(
            nn.Linear(4, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1)
        ).double()
        target = deepcopy(critic).requires_grad_(False)
        params = list(policy.parameters())
        opt_actor = torch.optim.Adam(params, lr=cfg.actor_lr)
        opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        decay = torch.optim.lr_scheduler.LinearLR(
            opt_actor, 1.0, 0.1, total_iters=cfg.iters
        )

        def restart():  # every plant: random wave phase, random buoy state
            obs = plant.reset(2 * np.pi * torch.rand(cfg.n_env, dtype=torch.float64))
            z = (
                cfg.restart_scale
                * (2 * torch.rand(cfg.n_env, 2, dtype=torch.float64) - 1)
                * scale[:2]
            )
            return torch.cat([z, obs[:, 2:]], dim=-1)

        obs = restart()
        steps, log = 0, []
        for it in range(-cfg.critic_warmup, cfg.iters):
            if steps >= cfg.episode_len:
                obs, steps = restart(), 0
            # The window starts here: the graph of earlier steps is cut.
            obs = obs.detach()
            obs_log, rews, ret, failed = [], [], 0.0, 0
            for t in range(
                cfg.horizon
            ):  # differentiable rollout of h steps, n plants at once
                obs_log.append(obs / scale)
                ctx, u = policy(obs)
                failed += int((ctx.status != 0).sum())
                obs, energy = plant.step(obs, u)
                rews.append(reward(energy, obs[:, 0], cfg))
                ret = ret + cfg.gamma**t * rews[-1]
            obs_log.append(obs / scale)
            # Bootstrap with the value of the last obs: stored energy plus target critic.
            ret = ret + cfg.gamma**cfg.horizon * (
                stored_energy(obs) + target(obs_log[-1]).squeeze(-1)
            )
            loss = -ret.mean() / cfg.horizon
            steps += cfg.horizon
            if it >= 0:  # warm-up iterations train the critic only
                opt_actor.zero_grad()
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
                opt_actor.step()
                decay.step()

            with torch.no_grad():  # regression targets from the same rollout
                O, R = torch.stack(obs_log), torch.stack(rews)
                Phi = stored_energy(O * scale)  # the critic learns the value minus Phi
                targets = td_lambda_targets(
                    R + cfg.gamma * Phi[1:] - Phi[:-1],  # shaped rewards r'
                    target(O[1:]).squeeze(-1),
                    cfg.gamma,
                    cfg.lam,
                )
                X, Yt = O[:-1].reshape(-1, 4), targets.reshape(-1)
            for _ in range(cfg.critic_epochs):
                critic_loss = torch.nn.functional.mse_loss(critic(X).squeeze(-1), Yt)
                opt_critic.zero_grad()
                critic_loss.backward()
                opt_critic.step()
            critic_loss = critic_loss.item()
            with torch.no_grad():  # Polyak update of the target critic
                for pt, pc in zip(target.parameters(), critic.parameters()):
                    pt.mul_(cfg.target_alpha).add_((1 - cfg.target_alpha) * pc)

            if it < 0:
                continue
            entry = {
                "it": it + 1,
                "env_steps": (cfg.critic_warmup + it + 1) * cfg.horizon * cfg.n_env,
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "critic_loss": critic_loss,
                "solver_failures": failed,
                "mean_reward": torch.stack(rews).mean().item(),
            }
            if callback is not None:
                entry.update(callback(it + 1, policy))
            log.append(entry)
        return log

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Train from the nominal $g = (1, 0)$ with the defaults of `ShacConfig`: 16 plants, $h = 10$,
    50 critic warm-up iterations and 150 actor updates. Every 10 iterations the MPC is evaluated
    with the closed loop of Section 3. About 20 s on a laptop CPU.
    """)
    return


@app.cell
def _(MpcPolicy, ShacConfig, X_MAX, mpc, plant, summarize, time, train):
    _cfg = ShacConfig(x_max=X_MAX)
    pol_shac = MpcPolicy(mpc)  # fresh start at g = (1, 0), the shared solver
    _start = summarize(pol_shac, plant)
    _t0 = time.time()
    shac_log = train(
        pol_shac,
        plant,
        _cfg,
        lambda it, pol: summarize(pol, plant) if it % 10 == 0 else {},
    )
    _wall = time.time() - _t0
    shac_log.insert(0, dict(it=0, env_steps=0, **_start))
    print(
        f"SHAC: {_cfg.iters} iterations, {shac_log[-1]['env_steps']} environment steps, {_wall:.0f} s, "
        f"{sum(e['solver_failures'] for e in shac_log[1:])} solver failures"
    )
    for _e in shac_log:
        if "power" in _e:
            print(
                f"   it {_e['it']:4d}  steps {_e['env_steps']:6d}  P_e {_e['power'] / 1e3:6.2f} kW  "
                f"max|x| {_e['xmax']:.3f} m  gain {_e['gain']:.3f}  phase {_e['phase']:5.1f} deg"
            )
    return pol_shac, shac_log


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Results

    Steady-state power of every controller on the plant, and how much of the gap between the
    nominal MPC and the oracle the training recovers.
    """)
    return


@app.cell
def _(G_TRUE, ev, mo, polar, result_table, shac_log):
    _nom, _orc, _pas = (
        ev["MPC, nominal model"],
        ev["MPC, true model (oracle)"],
        ev["passive damper"],
    )
    _fin = shac_log[-1]
    _rows = [
        ("MPC, nominal model", _nom["power"], _nom["xmax"], 1.0, 0.0),
        (
            "MPC trained by SHAC",
            _fin["power"],
            _fin["xmax"],
            _fin["gain"],
            _fin["phase"],
        ),
        ("MPC, true model (oracle)", _orc["power"], _orc["xmax"], *polar(G_TRUE)),
        ("passive damper", _pas["power"], _pas["xmax"], None, None),
    ]
    mo.md(
        result_table(_rows, _nom["power"], _orc["power"])
        + "\n\n(expected: SHAC about 33.5 kW, nearly all of the gap)"
    )
    return


@app.cell
def _(G_TRUE, ev, plot_training, shac_log):
    plot_training(
        shac_log,
        ev["MPC, nominal model"]["power"],
        ev["MPC, true model (oracle)"]["power"],
        G_TRUE,
    )
    return


@app.cell
def _(U_MAX, X_MAX, ev, evaluate, plant, plot_closed_loop, pol_shac):
    plot_closed_loop(
        {
            "MPC, nominal model": ev["MPC, nominal model"]["log"],
            "MPC trained by SHAC": evaluate(pol_shac, plant)["log"],
            "MPC, true model (oracle)": ev["MPC, true model (oracle)"]["log"],
        },
        X_MAX,
        U_MAX,
        plant.n_per,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
