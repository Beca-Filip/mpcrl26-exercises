import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import time
    from copy import deepcopy
    import numpy as np
    import torch
    from torch import nn
    from leap_c.parameters import AcadosParameterManager
    from leap_c.torch import AcadosDiffMpcTorch
    from utils import (
        C_BPTT, C_SHAC, C_SHORT, REWARD_SCALE, U_SCALE, LinearPolicy, RegularWaveWEC, ShacConfig,
        WECParams, design_lq, discretise, evaluate, g_polar, make_ocp, optimal_damper, phiw_powers,
        plot_closed_loop, plot_landscape, plot_phasors, plot_training, polar, power_landscape,
        result_table, summarize, wave_forecast,
    )

    return (
        AcadosDiffMpcTorch,
        AcadosParameterManager,
        C_BPTT,
        C_SHAC,
        C_SHORT,
        LinearPolicy,
        REWARD_SCALE,
        RegularWaveWEC,
        ShacConfig,
        U_SCALE,
        WECParams,
        deepcopy,
        design_lq,
        discretise,
        evaluate,
        g_polar,
        make_ocp,
        mo,
        nn,
        np,
        optimal_damper,
        phiw_powers,
        plot_closed_loop,
        plot_landscape,
        plot_phasors,
        plot_training,
        polar,
        power_landscape,
        result_table,
        summarize,
        time,
        torch,
        wave_forecast,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 10 — Learning an MPC through a differentiable simulator

    **Goal:** turn an acados MPC into a differentiable PyTorch policy with leap-c, write the
    plant as a differentiable torch function, and train a parameter of the MPC with the
    gradient of the simulated closed-loop return: short-horizon actor-critic (SHAC) and
    truncated back-propagation through time (BPTT).

    Requirements: `leap_c` (main branch, commit `217e1c8`, API `leap_c.torch` and
    `leap_c.parameters`), acados 0.5.3 with its `t_renderer` binary, CPU torch. The notebook
    compiles small acados solvers (about 2 s each) into temporary directories.

    ## 1. A wave energy converter

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
    the angle of $g$. The long-wave model has $g = (1, 0)$. This exercise is about what happens
    when the plant has a different $g$ than the controller believes, and how to learn it.

    **Power.** The PTO absorbs the mechanical power $-u\dot x$ and loses $r u^2$ in its
    windings, so the electrical power is $P_e = -u v - r u^2$ and the controller minimises the
    stage cost $\ell = u v + r u^2 = -P_e$. This cost is **indefinite**: an economic objective,
    not a tracking error.

    **Constraints.** A stroke limit $|x| \le 1.5$ m (soft: end stops, buoy must stay in the
    water) and a force limit $|u| \le 300$ kN (hard).

    **Simplifications.** Regular wave, hydrodynamic coefficients frozen at the wave frequency,
    full state information including the wave state $w$ (in reality $F_{ex}$ is estimated, which
    is where errors in $g$ come from), a smooth loss model, no viscous drag.

    **Discrete time.** With a zero-order hold on $u$ and $\Delta t = 0.2$ s the sampled-data
    model is exact: $z^+ = A z + G_w w + B_u u$ and $w^+ = \Phi_w w$ with $z = (x, v)$, and the
    electrical energy of one sample is the quadratic form $-y^\top W_d\, y$ of
    $y = (z, w, u)$. `discretise` in `utils.py` computes $A, B_u, G_w, \Phi_w, W_d$ with the Van
    Loan integral; the simulator uses the same matrices, so its energy accounting has no
    integration error.
    """)
    return


@app.cell
def _(RegularWaveWEC, WECParams, discretise, g_polar, np):
    DT, N, X_MAX, U_MAX = 0.2, 20, 1.5, 300e3            # sample time [s], MPC horizon, stroke [m], force [N]
    G_NOM, G_TRUE = np.array([1.0, 0.0]), g_polar(0.8, 40.0)   # the MPC's model and the plant
    GAINS, PHASES = np.linspace(0.3, 1.2, 10), np.linspace(-10, 70, 17)   # grid of the power landscape
    N_LAND = len(GAINS) * len(PHASES)
    p = WECParams()
    d = discretise(p, DT)                                # nominal model: buoy, wave and cost matrices
    env_nom = RegularWaveWEC(p, DT)                      # plant with the nominal excitation g = (1, 0)
    print(f"Inertia M = {p.M/1e3:.1f} t, stiffness k = {p.k/1e3:.0f} kN/m, radiation damping B = {p.B/1e3:.1f} kNs/m")
    print(f"Natural period {p.natural_period:.2f} s vs wave period {p.wave_period:.0f} s; "
          f"excitation force amplitude {p.F_amp/1e3:.0f} kN; r = {p.r:.2e} 1/(N s/m)")
    print(f"{env_nom.n_per} samples per wave period; Phiw (rotation by omega dt) =\n{d.Phiw}")
    return DT, GAINS, G_NOM, G_TRUE, N, N_LAND, PHASES, U_MAX, X_MAX, d, env_nom, p


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Baseline: a passive damper

    The simplest PTO control is a linear damper $u = -c\,v$, which can only absorb power. The
    best $c$ follows from the frequency domain (it maximises
    $\tfrac12 c\,(1 - r c)\,|\hat F_{ex}|^2 / |Z_i + c|^2$ with the buoy impedance
    $Z_i = B + j(\omega M - k/\omega)$) and is re-optimised for the sampled loop. It is not the
    focus here, only the reference that every other controller is compared with.
    """)
    return


@app.cell
def _(LinearPolicy, d, env_nom, evaluate, optimal_damper, p):
    c_pas = optimal_damper(d, p)
    ev_pas_nom = evaluate(LinearPolicy.damper(c_pas), env_nom)
    print(f"Damper c = {c_pas/1e3:.1f} kNs/m: P_e {ev_pas_nom['power']/1e3:.2f} kW, max|x| {ev_pas_nom['xmax']:.3f} m, "
          f"velocity leads F_ex by {ev_pas_nom['v_phase']:.1f} deg (expected 10.07 kW, 0.419 m)")
    return (ev_pas_nom,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Economic MPC with the nominal model

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

    $P$ solves a discrete algebraic Riccati equation and $Y$ a Stein equation (`design_lq`).
    The stroke limit is soft because a hard bound is not recursively feasible against a
    persistent wave. Note that the stage Hessian $W_d$ and $P$ are indefinite. The OCP is given
    (`make_ocp`): acados with discrete dynamics, external cost, HPIPM with the classical
    Riccati recursion, inputs scaled by $10^5$ N and costs by $10^3$ J.

    **Parameters in leap-c.** The OCP has two parameters. Both are registered with an
    `AcadosParameterManager`, which returns the CasADi symbols the OCP expressions are built
    from and later turns the values passed at solve time into acados parameters:

    | name | registered as | enters acados as | value passed to `params` |
    |---|---|---|---|
    | `g` | `differentiable=True` | `p_global`, sensitivities available | tensor of shape `(B, 2)` |
    | `wave` | default, non-differentiable | stage-wise `p` | numpy array of shape `(B, N+1, 2)` |

    A differentiable MPC layer needs at least one differentiable parameter (acados computes
    parametric sensitivities only for `p_global`), so `g` is declared differentiable now, even
    though this section holds it fixed. `AcadosDiffMpcTorch` generates and compiles the solver
    for a batch of `n_batch_init` problems that are solved in parallel.
    """)
    return


@app.cell
def _(
    AcadosDiffMpcTorch,
    AcadosParameterManager,
    G_NOM,
    N,
    N_LAND,
    U_MAX,
    X_MAX,
    d,
    make_ocp,
    np,
    phiw_powers,
    torch,
):
    manager_nom = AcadosParameterManager(N_horizon=N)
    g_sym = manager_nom.register_parameter("g", G_NOM, differentiable=True)   # excitation coefficient of the MPC model
    wave_sym = manager_nom.register_parameter("wave", np.zeros(2))           # wave state at every stage (forecast)
    ocp_nom = make_ocp(d, N, g_sym, wave_sym, x_max=X_MAX, u_max=U_MAX, name="wec_nominal")
    mpc_nom = AcadosDiffMpcTorch(ocp_nom, manager_nom, n_batch_init=N_LAND, dtype=torch.float64, verbose=False)
    phi_pow = phiw_powers(d, N)   # Phiw^k for k = 0..N, builds the forecast from the current w
    print(manager_nom)
    return mpc_nom, phi_pow


@app.cell
def _(U_SCALE, np, torch, wave_forecast):
    def mpc_act(mpc, g, phi_pow):
        """Policy (z, w) -> u [N] of an AcadosDiffMpcTorch for one state at a time and a fixed g."""
        g_t = torch.as_tensor(np.asarray(g, dtype=float))[None]   # differentiable parameter: tensor (B, 2), B = 1

        def act(z, w):
            with torch.no_grad():
                ctx, u0, x, u, value = mpc(torch.from_numpy(z)[None],
                                           params={"g": g_t, "wave": wave_forecast(phi_pow, w[None])})
            return float(u0[0, 0]) * U_SCALE

        return act

    return (mpc_act,)


@app.cell
def _(
    AcadosDiffMpcTorch,
    AcadosParameterManager,
    G_NOM,
    N,
    N_LAND,
    d,
    design_lq,
    make_ocp,
    mpc_act,
    mpc_nom,
    np,
    phi_pow,
    time,
    torch,
    wave_forecast,
):
    # Without active constraints the MPC must reproduce the LQ law u = -Kz z - Kw w for any N.
    _manager = AcadosParameterManager(N_horizon=N)
    _ocp = make_ocp(d, N, _manager.register_parameter("g", G_NOM, differentiable=True),
                    _manager.register_parameter("wave", np.zeros(2)), name="wec_unconstrained")
    _mpc = AcadosDiffMpcTorch(_ocp, _manager, n_batch_init=4, dtype=torch.float64, verbose=False)
    _lq = design_lq(d)
    _rng = np.random.default_rng(2)
    _z, _w = _rng.normal(size=(4, 2)), _rng.normal(size=(4, 2)) * 0.5
    _act = mpc_act(_mpc, G_NOM, phi_pow)
    _u_mpc = np.array([_act(_z[i], _w[i]) for i in range(4)])
    _u_lq = np.array([_lq.policy(_z[i], _w[i]) for i in range(4)])
    print(f"Unconstrained MPC vs LQ law, relative difference (expected < 1e-8): "
          f"{np.abs(_u_mpc - _u_lq).max() / np.abs(_u_lq).max():.1e}")
    print(f"H = R + B'PB = {_lq.H:.2e} > 0, DARE residual {_lq.dare_residual:.1e}, "
          f"closed-loop |eig| = {np.abs(_lq.cl_eigs).max():.3f}")
    _act = mpc_act(mpc_nom, G_NOM, phi_pow)
    _t0 = time.perf_counter()
    for _ in range(200):
        _act(_z[0], _w[0])
    _single = (time.perf_counter() - _t0) / 200
    _zb, _wb = _rng.normal(size=(N_LAND, 2)), _rng.normal(size=(N_LAND, 2)) * 0.4
    _t0 = time.perf_counter()
    with torch.no_grad():
        mpc_nom(torch.from_numpy(_zb), params={"g": torch.as_tensor(np.tile(G_NOM, (N_LAND, 1))),
                                               "wave": wave_forecast(phi_pow, _wb)})
    _batch = (time.perf_counter() - _t0) / N_LAND
    print(f"Solve time: {1e3*_single:.2f} ms for one OCP, {1e3*_batch:.3f} ms per OCP in a batch of {N_LAND}")
    return


@app.cell
def _(
    G_NOM,
    U_MAX,
    X_MAX,
    d,
    design_lq,
    env_nom,
    ev_pas_nom,
    evaluate,
    mpc_act,
    mpc_nom,
    phi_pow,
    plot_closed_loop,
):
    act_nom = mpc_act(mpc_nom, G_NOM, phi_pow)
    _ev = {"constrained MPC": evaluate(act_nom, env_nom),
           "LQ, unconstrained": evaluate(design_lq(d).policy, env_nom),
           "passive damper": ev_pas_nom}
    for _name, _e in _ev.items():
        print(f"{_name:18s} P_e {_e['power']/1e3:6.2f} kW  max|x| {_e['xmax']:.3f} m  max|u| {_e['umax']/1e3:4.0f} kN  "
              f"v leads F_ex by {_e['v_phase']:5.1f} deg")
    print("(expected: MPC 46.37 kW, 1.500 m, 300 kN; LQ 60.74 kW, 3.159 m, 506 kN)")
    plot_closed_loop({k: v["log"] for k, v in _ev.items()}, X_MAX, U_MAX, env_nom.n_per)
    return (act_nom,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question:** every stage Hessian $W_d$ and the terminal $P$ are indefinite, yet the QP is
    convex and HPIPM solves it without regularisation. Why? And what does the MPC do while no
    constraint is active?
    """).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The same MPC on a mismatched plant

    Now the plant's excitation force is 20 % weaker and lags the wave by 40 degrees:
    $g^\star = 0.8\,(\cos 40^\circ, \sin 40^\circ)$. `RegularWaveWEC(..., g_ex=G_TRUE)` applies it
    exactly, because $G(g)$ commutes with the rotation $\Phi_w$. The MPC still believes
    $g = (1, 0)$. Evaluation is always the same: 20 periods from rest, numbers averaged over the
    last five. For reference, the oracle is the same MPC with the true $g$ in its model.
    """)
    return


@app.cell
def _(
    DT,
    G_TRUE,
    LinearPolicy,
    RegularWaveWEC,
    act_nom,
    evaluate,
    mpc_act,
    mpc_nom,
    optimal_damper,
    p,
    phi_pow,
):
    env_true = RegularWaveWEC(p, DT, g_ex=G_TRUE)        # the plant: g differs from the MPC's (1, 0)
    ev_true = {"MPC, nominal model": evaluate(act_nom, env_true),
               "MPC, true model (oracle)": evaluate(mpc_act(mpc_nom, G_TRUE, phi_pow), env_true),
               "passive damper": evaluate(LinearPolicy.damper(optimal_damper(env_true.d, p)), env_true)}
    for _name, _e in ev_true.items():
        print(f"{_name:26s} P_e {_e['power']/1e3:6.2f} kW  max|x| {_e['xmax']:.3f} m  v leads F_ex by {_e['v_phase']:5.1f} deg")
    print("(expected: nominal 25.26 kW, 1.519 m; oracle 33.56 kW, 1.500 m; damper 6.44 kW)")
    return env_true, ev_true


@app.cell
def _(U_MAX, X_MAX, env_true, ev_true, mo, plot_closed_loop, plot_phasors):
    _logs = {k: v["log"] for k, v in ev_true.items()}
    mo.vstack([plot_closed_loop(_logs, X_MAX, U_MAX, env_true.n_per), plot_phasors(_logs, env_true.n_per)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How bad can a wrong $g$ be? The closed-loop power of the MPC on the true plant as a function
    of the gain and phase of *its* excitation coefficient. All 170 grid points are simulated as
    one batch of plants (about 80 000 OCP solves). An occasional `ACADOS_MINSTEP` line is
    HPIPM giving up on a step length in one of them; the solve still returns a usable iterate.
    """)
    return


@app.cell
def _(mo):
    run_land = mo.ui.run_button(label="Compute the power landscape (about 15 s)")
    run_land
    return (run_land,)


@app.cell
def _(
    GAINS,
    G_TRUE,
    PHASES,
    env_true,
    mo,
    mpc_nom,
    np,
    phi_pow,
    plot_landscape,
    power_landscape,
    run_land,
):
    mo.stop(not run_land.value)
    land = power_landscape(mpc_nom, phi_pow, env_true, GAINS, PHASES)
    _k = np.unravel_index(land["P"].argmax(), land["P"].shape)
    _i_nom, _j_nom = np.argmin(abs(land["gain"] - 1)), np.argmin(abs(land["phase"]))
    print(f"Grid maximum {land['P'].max()/1e3:.2f} kW at gain {land['gain'][_k[0]]:.2f}, phase {land['phase'][_k[1]]:.0f} deg "
          f"(true coefficient 0.80, 40 deg); at the nominal model {land['P'][_i_nom, _j_nom]/1e3:.2f} kW")
    plot_landscape(land, G_TRUE)
    return (land,)


@app.cell
def _(mo):
    gain_s = mo.ui.slider(0.3, 1.2, step=0.05, value=1.0, label="MPC excitation gain")
    phase_s = mo.ui.slider(-10, 70, step=5, value=0, label="MPC excitation phase lag [deg]")
    run_try = mo.ui.run_button(label="Evaluate on the true plant (800 solves)")
    mo.vstack([mo.hstack([gain_s, phase_s]), run_try])
    return gain_s, phase_s, run_try


@app.cell
def _(env_true, evaluate, g_polar, gain_s, mo, mpc_act, mpc_nom, phase_s, phi_pow, run_try):
    mo.stop(not run_try.value)
    # g is a runtime parameter of the compiled solver: no recompilation when it changes.
    _ev = evaluate(mpc_act(mpc_nom, g_polar(gain_s.value, phase_s.value), phi_pow), env_true)
    print(f"MPC with gain {gain_s.value:.2f}, phase {phase_s.value:.0f} deg on the true plant: "
          f"P_e {_ev['power']/1e3:.2f} kW, max|x| {_ev['xmax']:.3f} m, v leads F_ex by {_ev['v_phase']:.1f} deg")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question:** the nominal MPC overshoots the stroke limit by only 2 cm, yet it loses a
    quarter of the power. Where does the power go, and why does the landscape fall off much
    faster along the phase axis than along the gain axis?
    """).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. A differentiable plant and a differentiable MPC

    Why would $g$ be wrong? The excitation force cannot be measured. It is reconstructed from a
    wave estimate (a sensor on the buoy, a wave rider some distance up-wave, a forecast), its
    amplitude and phase relative to the wave elevation depend on the diffraction correction of
    the hydrodynamic model, and the buoy itself moves. At one wave frequency all of these errors
    are a single complex number, a gain and a phase: $g$. We cannot measure $g$, but we can
    measure what counts, the energy the plant delivers under the controller. So treat $g$ as a
    **policy parameter** and improve it by gradient ascent on the simulated closed-loop energy.
    That needs two things:

    1. the plant as a differentiable function in PyTorch, batched over many plants that run in
       parallel;
    2. the MPC as a differentiable function of its parameter $g$ **and** of its initial state
       $z_0$: the gradient must flow through the state from one step to the next. leap-c
       provides $\partial u_0/\partial g$ and $\partial u_0/\partial z_0$ from the implicit
       function theorem on the KKT conditions of the converged QP, reusing HPIPM's factorisation
       (Frey et al. 2025).

    ### The plant in torch

    `DiffWEC` is the simulator of `utils.py` written as pure tensor functions: it stores no
    state, the caller passes $(z, w)$ in and receives $(z^+, w^+, e)$, and the leading dimension
    is a free batch of plants. Autograd then records the whole rollout. Complete `step`: the
    electrical energy of the sample is $e = -y^\top W_d\, y$ with $y = (z, w, u)$, and
    $z^+ = A z + G_w w + B_u u$. Mind the batch: rows of `z`, `w` are plants, so the matrices
    multiply from the right, transposed.
    """)
    return


@app.cell
def _(nn, torch):
    class DiffWEC(nn.Module):
        """Plant z' = A z + Gw w + Bu u, w' = Phiw w with the exact step energy -y' Wd y, batched."""

        def __init__(self, d, amp):
            super().__init__()
            for name in ("A", "Gw", "Phiw", "Wd"):
                self.register_buffer(name, torch.from_numpy(getattr(d, name)))
            self.register_buffer("Bu", torch.from_numpy(d.Bu[:, 0]))
            self.amp, self.dt = amp, d.dt

        def reset(self, phase):
            """Buoys at rest, waves eta = A cos(omega t + phase) for a batch of phases (B,). Returns z, w: (B, 2)."""
            w = self.amp * torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)
            return torch.zeros_like(w), w

        def step(self, z, w, u):
            """z, w: (B, 2), u: (B,) [N]. Returns z', w' and the electrical energy of the step (B,) [J]."""
            y = torch.cat([z, w, u[:, None]], dim=-1)
            energy = ...  # TODO: electrical energy of the step, -y' Wd y, one scalar per plant (torch.einsum).
            z_next = ...  # TODO: A z + Gw w + Bu u for the whole batch.
            w_next = w @ self.Phiw.T
            return z_next, w_next, energy

    return (DiffWEC,)


@app.cell
def _(DiffWEC, env_true, np, p, torch):
    plant = DiffWEC(env_true.d, p.amp)   # the true plant, in torch
    _rng = np.random.default_rng(0)
    _z, _w = env_true.reset()
    _zt, _wt = torch.from_numpy(_z)[None], torch.from_numpy(_w)[None]
    _err, _err_e = 0.0, 0.0
    for _ in range(50):
        _u = float(_rng.normal() * 2e5)
        _z, _w, _e = env_true.step(_u)
        _zt, _wt, _et = plant.step(_zt, _wt, torch.tensor([_u], dtype=torch.float64))   # the plant is float64 throughout
        _err = max(_err, np.abs(_zt[0].numpy() - _z).max(), np.abs(_wt[0].numpy() - _w).max())
        _err_e = max(_err_e, abs(_et.item() - _e) / abs(_e))
    print(f"DiffWEC vs numpy simulator over 50 random steps: state error {_err:.1e} (expected < 1e-12), "
          f"energy relative error {_err_e:.1e} (expected < 1e-10)")
    return (plant,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The MPC as a torch module

    `MpcPolicy` wraps the acados solver of Section 3 as an `nn.Module` whose only trainable
    parameter is `g`. Three details matter:

    * The solver sees `g` through the `params` dict. Pass `self.g` itself, expanded to the
      batch, not a copy: autograd must see the path from the parameter into the solver.
    * The forecast `wave` is a non-differentiable parameter and is built from the current wave
      state with `w.detach()`. The wave is exogenous, nothing the policy does changes it, and
      leap-c raises a `ValueError` if a tensor that requires grad is passed for a
      non-differentiable parameter.
    * Do not pass leap-c's `discount_factor`: it rescales the stage costs by $\gamma^k$ and
      would silently change this economic OCP. (`make_ocp` sets `cost_scaling` itself, because
      acados otherwise multiplies stage costs by $\Delta t$.)

    Every `MpcPolicy` compiles its own solver, so give each instance a distinct `name`;
    `n_batch` must cover the number of plants trained in parallel. Complete `forward`.
    """)
    return


@app.cell
def _(
    AcadosDiffMpcTorch,
    AcadosParameterManager,
    U_SCALE,
    make_ocp,
    nn,
    np,
    phiw_powers,
    torch,
    wave_forecast,
):
    class MpcPolicy(nn.Module):
        """u = MPC_g(z, w) [N] for a batch of states, differentiable in g and in z."""

        def __init__(self, d, N, x_max, u_max, g_init=(1.0, 0.0), n_batch=16, name="wec_diff"):
            super().__init__()
            self.g = nn.Parameter(torch.as_tensor(np.asarray(g_init, dtype=float)))   # the learnable excitation coefficient
            manager = AcadosParameterManager(N_horizon=N)
            g = manager.register_parameter("g", np.asarray(g_init, dtype=float), differentiable=True)
            wave = manager.register_parameter("wave", np.zeros(2))
            ocp = make_ocp(d, N, g, wave, x_max=x_max, u_max=u_max, name=name)
            self.diff_mpc = AcadosDiffMpcTorch(ocp, manager, n_batch_init=n_batch, dtype=torch.float64, verbose=False)
            self._phi_pow = phiw_powers(d, N)

        def forward(self, z, w, ctx=None):
            """z, w: (B, 2). Returns the PTO force (B,) [N] and the leap-c context (solver status, warm start)."""
            forecast = wave_forecast(self._phi_pow, w.detach().numpy())   # exogenous: no gradient through the sea
            params = ...  # TODO: {"g": self.g for every plant, shape (B, 2), graph intact, "wave": forecast}.
            ctx, u0, x, u, value = self.diff_mpc(z, params=params, ctx=ctx)
            return u0[:, 0] * U_SCALE, ctx

        @torch.no_grad()
        def act(self, z, w):
            """One numpy state in, one force out: the interface of `evaluate`."""
            return float(self(torch.from_numpy(z)[None], torch.from_numpy(w)[None])[0])

    return (MpcPolicy,)


@app.cell
def _(MpcPolicy, N, U_MAX, X_MAX, act_nom, d, np):
    policy = MpcPolicy(d, N, X_MAX, U_MAX, n_batch=8, name="wec_diff")
    _rng = np.random.default_rng(1)
    _z, _w = _rng.normal(size=(8, 2)) * 0.8, _rng.normal(size=(8, 2)) * 0.4
    _z[:4, 0] = _rng.choice([-1, 1], size=4) * _rng.uniform(1.3, 1.5, size=4)   # near the stroke limit
    _du = max(abs(policy.act(_z[i], _w[i]) - act_nom(_z[i], _w[i])) for i in range(8))
    print(f"Trainable parameters: {sum(q.numel() for q in policy.parameters())} (expected 2)")
    print(f"MpcPolicy vs the Section 3 solver on 8 states, max |du| = {_du:.1e} N (expected < 1e-6)")
    return (policy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is it really differentiable?

    Roll the policy out for 20 steps on 4 plants, sum the energy, and compare the gradient with
    respect to $g$ from autograd with central finite differences. Half of the initial states sit
    near the stroke limit so that constraints are active along the way.
    """)
    return


@app.cell
def _(REWARD_SCALE, np, plant, policy, torch):
    def rollout(policy, plant, z0, w0, steps=20):
        """Energy of a closed-loop rollout from (z0, w0), summed over plants and steps, in units of REWARD_SCALE."""
        z, w, R = z0, w0, 0.0
        for _ in range(steps):
            u, ctx = policy(z, w)
            z, w, e = plant.step(z, w, u)
            R = R + e.sum() / REWARD_SCALE
        return R

    _rng = np.random.default_rng(3)
    _z0, _w0 = _rng.normal(size=(4, 2)) * 0.8, _rng.normal(size=(4, 2)) * 0.4
    _z0[:2, 0] = _rng.choice([-1, 1], size=2) * _rng.uniform(1.3, 1.5, size=2)
    _z0, _w0 = torch.from_numpy(_z0), torch.from_numpy(_w0)
    _g0 = torch.tensor([0.9, 0.2], dtype=torch.float64)
    with torch.no_grad():
        policy.g.copy_(_g0)
    _grad = ...  # TODO: gradient of rollout(policy, plant, _z0, _w0) with respect to policy.g (torch.autograd.grad).
    _fd = np.zeros(2)
    for _i in range(2):
        _vals = []
        for _s in (1, -1):
            with torch.no_grad():
                policy.g.copy_(_g0)
                policy.g[_i] += _s * 1e-5
            _vals.append(rollout(policy, plant, _z0, _w0).item())
        _fd[_i] = (_vals[0] - _vals[1]) / 2e-5
    with torch.no_grad():
        policy.g.copy_(torch.tensor([1.0, 0.0]))   # back to the nominal model
    print(f"dR/dg autograd {_grad.numpy()}, finite differences {_fd}, "
          f"relative error {np.abs(_grad.numpy() - _fd).max() / np.abs(_fd).max():.1e} (expected < 1e-3)")
    return (rollout,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question:** why is the forecast built from `w.detach()` rather than from `w`, and why
    does autograd agree with finite differences although the MPC, with its active-set changes,
    is only piecewise smooth?
    """).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Learning with SHAC and truncated BPTT

    **Objective.** Maximise the discounted closed-loop return of $n$ plants started from random
    wave phases and random buoy states,

    $$
    J(g) = \mathbb{E}\Big[\sum_{t \ge 0} \gamma^t r_t\Big],\qquad
    r_t = \frac{1}{10^5\,\mathrm{J}}\Big(e_t - \rho\,\operatorname{relu}\big(|x_{t+1}| - x_{\max}\big)^2\Big),
    $$

    where $e_t$ is the electrical energy of step $t$ from `plant.step` and the penalty with
    $\rho = 10^8$ J/m² keeps the learned controller inside the stroke that the MPC's soft
    constraint promises.

    **Pathwise gradient over a short window.** Simulator and policy are differentiable, so the
    gradient of a window of $h$ steps is one `backward()` through $h$ plant steps and $h$ MPC
    solves: no action noise, no likelihood ratio. The loss of one iteration is

    $$
    L(g) = -\frac{1}{h}\,\frac{1}{n}\sum_{i=1}^{n}\Big[\sum_{t=0}^{h-1}\gamma^t r_t^{(i)}
      + \gamma^h\, V_\phi\big(s_h^{(i)}\big)\Big].
    $$

    Without the last term this is **truncated back-propagation through time**: the plants keep
    running across iterations, but the graph is cut every $h$ steps with `detach`. The bias of
    the truncation is the value of the state the window leaves behind. Here that is the energy
    stored in the moving buoy, $\tfrac12 k x^2 + \tfrac12 M v^2 \approx 200$ kJ, against a net
    yield of about 50 kJ per step: a 10-step window is rewarded for *not* storing energy, which
    means a small excitation gain.

    **SHAC** (short-horizon actor-critic, Xu et al. 2022) closes the window with a learned
    value $V_\phi(s_h)$. The critic is a small MLP on the scaled state $s = (z, w)$, fitted by
    regression to TD($\lambda$) returns of the same rollouts,

    $$
    G_t = r_t + \gamma\big[(1 - \lambda)\, V_{\bar\phi}(s_{t+1}) + \lambda\, G_{t+1}\big],
    \qquad G_h = V_{\bar\phi}(s_h),
    $$

    with a slowly moving target network $\bar\phi$. An untrained critic would leave exactly the
    truncated gradient, so the first `critic_warmup` iterations fit the critic only.

    **Batching and exploration.** The $n$ plants are one batch of the acados solver
    (`n_batch = cfg.n_env`): one call `policy(z, w)` solves $n$ OCPs in parallel. There is no
    action noise. Diversity comes from the random wave phases and buoy states at the restarts
    every `episode_len` steps, which also keep the critic trained on states off the current
    limit cycle. Gaussian noise on the force was tried and made SHAC collapse: it adds $r u^2$
    losses and noise to the critic's targets without informing the gradient.

    `ShacConfig` in `utils.py` collects the settings. Complete `reward`, `td_lambda_targets` and
    the two marked lines of `train`, then read the loop: it is the whole pipeline.
    """)
    return


@app.cell
def _(REWARD_SCALE, torch):
    def reward(energy, x_next, cfg):
        """Electrical energy of the step minus the stroke-violation penalty, in units of REWARD_SCALE. All (B,)."""
        return ...  # TODO: energy minus the quadratic penalty on |x_next| beyond cfg.x_max, in units of REWARD_SCALE.

    return (reward,)


@app.cell
def _(ShacConfig, reward, torch):
    print("Reward for 100 kJ at x = 1.0 m and at x = 1.6 m (expected [1.0, -9.0]):",
          reward(torch.tensor([1e5, 1e5]), torch.tensor([1.0, 1.6]), ShacConfig(x_max=1.5)))
    return


@app.cell
def _(torch):
    def td_lambda_targets(R, v_next, gamma, lam):
        """TD(lambda) returns G_t for t = 0..h-1. R: rewards (h, B); v_next: target values V(s_{t+1}), (h, B)."""
        G, targets = v_next[-1], []
        for t in reversed(range(len(R))):
            G = ...  # TODO: the TD(lambda) recursion of the formula above.
            targets.append(G)
        return torch.stack(targets[::-1])

    return (td_lambda_targets,)


@app.cell
def _(td_lambda_targets, torch):
    print("TD(lambda) targets (expected [[2.125], [2.5]]):",
          td_lambda_targets(torch.tensor([[1.0], [1.0]]), torch.tensor([[2.0], [3.0]]), gamma=0.5, lam=0.5))
    return


@app.cell
def _(deepcopy, nn, np, reward, td_lambda_targets, torch):
    def train(policy, plant, cfg, callback=None):
        """SHAC (cfg.critic) or truncated BPTT. Trains policy.parameters() in place; returns a log per iteration."""
        torch.manual_seed(cfg.seed)
        omega = float(torch.atan2(plant.Phiw[1, 0], plant.Phiw[0, 0])) / plant.dt
        scale = torch.tensor([cfg.x_max, omega * cfg.x_max, plant.amp, plant.amp])   # critic input scaling of s = (z, w)
        critic = nn.Sequential(nn.Linear(4, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1)).double()
        target = deepcopy(critic).requires_grad_(False)
        params = list(policy.parameters())
        opt_actor = torch.optim.Adam(params, lr=cfg.actor_lr)
        opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        decay = torch.optim.lr_scheduler.LinearLR(opt_actor, 1.0, 0.1, total_iters=cfg.iters)
        warmup = cfg.critic_warmup if cfg.critic else 0

        def restart():   # every plant: random wave phase, random buoy state
            _, w = plant.reset(2 * np.pi * torch.rand(cfg.n_env, dtype=torch.float64))
            z = cfg.restart_scale * (2 * torch.rand(cfg.n_env, 2, dtype=torch.float64) - 1) * scale[:2]
            return z, w

        z, w = restart()
        steps, log = 0, []
        for it in range(-warmup, cfg.iters):
            if steps >= cfg.episode_len:
                (z, w), steps = restart(), 0
            z, w = ...  # TODO: cut the graph at the window boundary (truncation).
            obs, rews, ret, failed = [], [], 0.0, 0
            for t in range(cfg.horizon):   # differentiable rollout of h steps, n plants at once
                obs.append(torch.cat([z, w], dim=-1) / scale)
                u, ctx = policy(z, w)
                failed += int((ctx.status != 0).sum())
                if cfg.action_noise > 0:
                    u = u + cfg.action_noise * torch.randn_like(u)
                z, w, energy = plant.step(z, w, u)
                rews.append(reward(energy, z[:, 0], cfg))
                ret = ret + cfg.gamma**t * rews[-1]
            obs.append(torch.cat([z, w], dim=-1) / scale)
            if cfg.critic:
                ret = ret + ...  # TODO: bootstrap the window with the target critic's value of the last observation.
            loss = -ret.mean() / cfg.horizon
            steps += cfg.horizon
            if it >= 0:   # warm-up iterations train the critic only
                opt_actor.zero_grad()
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
                opt_actor.step()
                decay.step()

            critic_loss = float("nan")
            if cfg.critic:
                with torch.no_grad():   # regression targets from the same rollout
                    O, R = torch.stack(obs), torch.stack(rews)
                    targets = td_lambda_targets(R, target(O[1:]).squeeze(-1), cfg.gamma, cfg.lam)
                    X, Yt = O[:-1].reshape(-1, 4), targets.reshape(-1)
                for _ in range(cfg.critic_epochs):
                    critic_loss = torch.nn.functional.mse_loss(critic(X).squeeze(-1), Yt)
                    opt_critic.zero_grad()
                    critic_loss.backward()
                    opt_critic.step()
                critic_loss = critic_loss.item()
                with torch.no_grad():   # Polyak update of the target critic
                    for pt, pc in zip(target.parameters(), critic.parameters()):
                        pt.mul_(cfg.target_alpha).add_((1 - cfg.target_alpha) * pc)

            if it < 0:
                continue
            entry = dict(it=it + 1, env_steps=(warmup + it + 1) * cfg.horizon * cfg.n_env, loss=loss.item(),
                         grad_norm=grad_norm, critic_loss=critic_loss, solver_failures=failed,
                         mean_reward=torch.stack(rews).mean().item())
            if callback is not None:
                entry.update(callback(it + 1, policy))
            log.append(entry)
        return log

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question:** DDPG in Exercise 06 needed exploration noise on the action. Why does this
    method not, and what plays the role of exploration here?
    """).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Three runs: SHAC with a 10-step window, long-window BPTT without a critic as the
    reference, and the same 10-step window without a critic. Every run starts from the nominal
    $g = (1, 0)$ and is evaluated every 10 iterations with the 20-period closed loop of
    Section 4. About 30 s per run on a laptop CPU; the iteration budget can be reduced.
    """)
    return


@app.cell
def _(C_BPTT, C_SHAC, C_SHORT, mo):
    RUNS = {   # name -> (label, colour, ShacConfig overrides)
        "shac": ("SHAC, h = 10, critic", C_SHAC, dict(horizon=10, n_env=16, iters=300, actor_lr=0.01)),
        "bptt": ("BPTT, h = 80, no critic", C_BPTT, dict(horizon=80, n_env=8, iters=120, critic=False, gamma=1.0)),
        "short": ("truncated BPTT, h = 10, no critic", C_SHORT, dict(horizon=10, n_env=16, iters=300, actor_lr=0.01, critic=False)),
    }
    selected = mo.ui.multiselect(options=list(RUNS), value=list(RUNS), label="Runs")
    budget = mo.ui.slider(0.25, 1.0, step=0.25, value=1.0, label="Iteration budget (fraction of the default)")
    run_train = mo.ui.run_button(label="Train")
    mo.vstack([mo.hstack([selected, budget]), run_train])
    return RUNS, budget, run_train, selected


@app.cell
def _(
    MpcPolicy,
    N,
    RUNS,
    ShacConfig,
    U_MAX,
    X_MAX,
    budget,
    d,
    env_true,
    mo,
    plant,
    run_train,
    selected,
    summarize,
    time,
    train,
):
    mo.stop(not run_train.value)
    runs = {}
    for _name in selected.value:
        _label, _color, _over = RUNS[_name]
        _cfg = ShacConfig(x_max=X_MAX, **dict(_over, iters=max(10, int(budget.value * _over["iters"]))))
        _pol = MpcPolicy(d, N, X_MAX, U_MAX, n_batch=_cfg.n_env, name=f"wec_{_name}")   # fresh start at g = (1, 0)
        _start = summarize(_pol, env_true)
        _t0 = time.time()
        _log = train(_pol, plant, _cfg, lambda it, pol: summarize(pol, env_true) if it % 10 == 0 else {})
        _wall = time.time() - _t0
        _log.insert(0, dict(it=0, env_steps=0, **_start))
        runs[_name] = dict(label=_label, color=_color, log=_log, final=summarize(_pol, env_true), policy=_pol,
                           wall=_wall, failures=sum(e["solver_failures"] for e in _log[1:]))
        print(f"-- {_label}: {_cfg.iters} iterations, {_log[-1]['env_steps']} environment steps, "
              f"{_wall:.0f} s, {runs[_name]['failures']} solver failures")
        for _e in _log:
            if "power" in _e:
                print(f"   it {_e['it']:4d}  steps {_e['env_steps']:6d}  P_e {_e['power']/1e3:6.2f} kW  "
                      f"max|x| {_e['xmax']:.3f} m  gain {_e['gain']:.3f}  phase {_e['phase']:5.1f} deg")
    return (runs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Results

    Steady-state power of every controller on the true plant, and how much of the gap between
    the nominal MPC and the oracle each run recovers. The landscape figure needs the button of
    Section 4 to have been pressed once.
    """)
    return


@app.cell
def _(G_TRUE, ev_true, mo, polar, result_table, runs):
    _rows = [("MPC, nominal model", ev_true["MPC, nominal model"]["power"], ev_true["MPC, nominal model"]["xmax"], 1.0, 0.0)]
    _rows += [(f"MPC trained by {r['label']}", r["final"]["power"], r["final"]["xmax"], r["final"]["gain"], r["final"]["phase"])
              for r in runs.values()]
    _rows += [("MPC, true model (oracle)", ev_true["MPC, true model (oracle)"]["power"], ev_true["MPC, true model (oracle)"]["xmax"], *polar(G_TRUE)),
              ("passive damper", ev_true["passive damper"]["power"], ev_true["passive damper"]["xmax"], None, None)]
    mo.md(result_table(_rows, ev_true["MPC, nominal model"]["power"], ev_true["MPC, true model (oracle)"]["power"])
          + "\n\n(expected: SHAC about 32.7 kW and 90 %, BPTT 33.5 kW and 99 %, truncated BPTT collapses to about 8.6 kW)")
    return


@app.cell
def _(G_TRUE, ev_true, land, plot_training, runs):
    plot_training(runs, ev_true["MPC, nominal model"]["power"], ev_true["MPC, true model (oracle)"]["power"], G_TRUE, land)
    return


@app.cell
def _(U_MAX, X_MAX, env_true, ev_true, evaluate, mo, plot_closed_loop, plot_phasors, runs):
    _best = max(runs, key=lambda k: runs[k]["final"]["power"])
    _logs = {"MPC, nominal model": ev_true["MPC, nominal model"]["log"],
             f"trained by {runs[_best]['label']}": evaluate(runs[_best]["policy"].act, env_true)["log"],
             "MPC, true model (oracle)": ev_true["MPC, true model (oracle)"]["log"]}
    mo.vstack([plot_closed_loop(_logs, X_MAX, U_MAX, env_true.n_per), plot_phasors(_logs, env_true.n_per)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question:** SHAC ends near gain 0.48 and BPTT near 0.66, while the plant has 0.8, yet
    both recover 90 to 99 % of the power. Has the learning identified the plant?
    """).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question:** truncated BPTT with $h = 10$ follows the right direction for about 70
    iterations and then collapses to $g \approx 0$. What is the mechanism, and which two
    ingredients keep SHAC's equally short window honest?
    """).callout(kind="info")
    return


if __name__ == "__main__":
    app.run()
