import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import casadi as ca
    import numpy as np
    import matplotlib.pyplot as plt

    return ca, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 01 — Dynamics and simulation

    **Goal:** build, discretize, differentiate, and simulate a pendulum model
    with [CasADi](https://web.casadi.org/docs/).

    A torque-controlled pendulum is a simple nonlinear system for studying
    how physical dynamics can be simulated and, later, controlled.
    We use the normalized dynamics

    $$
    \dot\theta=\omega,\qquad \dot\omega=\sin\theta+u.
    $$

    The angle is measured from the upright equilibrium; the state is
    $s=(\theta,\omega)$ and the input is torque $u$.
    Angular velocity changes the angle, while gravity and the applied torque
    change the angular velocity.

    ## 1. Symbolic dynamics
    The right-hand side $f(s,u)$ of the ODE $\dot s=f(s,u)$ is called its
    **vector field**: it specifies the rate of change of the state for each
    state and input.
    Define the two components of the vector field. What distinguishes an
    `SX` expression from a `Function`? Evaluate your function at $(\pi/2,0)$
    with zero torque. What shape and type does it return?

    Use the cells below to inspect the expression, construct a callable
    function, and try numerical inputs before moving on to discretization.
    """)
    return


@app.cell
def _(ca):
    s = ca.SX.sym("s", 2)
    u = ca.SX.sym("u")

    def vector_field(s, u):
        theta_dot = ...  # TODO: angle derivative.
        omega_dot = ...  # TODO: angular acceleration.
        return ca.vertcat(theta_dot, omega_dot)

    return s, u, vector_field


@app.cell
def _(s, u, vector_field):
    rhs = vector_field(s, u)
    rhs
    return (rhs,)


@app.cell
def _(ca, rhs, s, u):
    f = ca.Function("f", [s, u], [rhs])
    f
    return (f,)


@app.cell
def _(f, np):
    # Change these inputs to explore the continuous-time model.
    probe_state = np.array([np.pi / 2, 0.0])
    probe_input = 0.0
    rate = f(probe_state, probe_input)
    print("Return type:", type(rate), "shape:", rate.shape)
    print("As a NumPy array:", np.asarray(rate))
    rate
    return probe_input, probe_state, rate


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Discretization
    Complete classical RK4. For a substep $h$:

    $$
    \begin{aligned}
    k_1&=f(s,u), & k_2&=f(s+hk_1/2,u),\\
    k_3&=f(s+hk_2/2,u), & k_4&=f(s+hk_3,u),\\
    s^+&=s+h(k_1+2k_2+2k_3+k_4)/6.
    \end{aligned}
    $$

    Use five substeps per sampling interval $\Delta t=0.1$ and hold the
    input constant. Return a symbolic expression so CasADi can differentiate it.
    Build the transition function and evaluate one step before attempting a
    whole trajectory. Does reducing the sampling interval reduce the change
    from the initial state?
    """)
    return


@app.cell
def _(ca, vector_field):
    def rk4(s, u, dt, substeps=5):
        f = ca.Function("f", [s, u], [vector_field(s, u)])
        h = dt / substeps
        x = s
        for _ in range(substeps):
            # The input is held constant throughout all four slope evaluations.
            k1 = f(x, u)
            k2 = ...  # TODO: slope at the first midpoint estimate.
            k3 = ...  # TODO: slope at the second midpoint estimate.
            k4 = ...  # TODO: slope at the endpoint estimate.
            x = ...  # TODO: weighted RK4 update.
        return x

    return (rk4,)


@app.cell
def _(ca, rk4, s, u):
    dt = 0.1
    substeps = 5
    discrete_dynamics = rk4(s, u, dt, substeps)
    F = ca.Function("F", [s, u], [discrete_dynamics])
    F
    return F, discrete_dynamics, dt, substeps


@app.cell
def _(F, np, probe_input, probe_state):
    next_state = np.asarray(F(probe_state, probe_input)).ravel()
    print("Initial state:", probe_state)
    print("State after one interval:", next_state)
    next_state
    return (next_state,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Automatic differentiation
    Refer to CasADi's [calculus and algorithmic differentiation documentation](https://web.casadi.org/docs/#calculus-algorithmic-differentiation)
    for `jacobian` and related operations.

    Construct functions for $\partial f/\partial s$ and $\partial f/\partial u$.
    Evaluate them at $(s,u)=(0,0)$, convert the results to NumPy, and inspect
    their sparsity. Is the upright equilibrium stable without feedback?
    Also differentiate the **discrete** dynamics and compare with
    $A_d\approx I+\Delta t A_c$, $B_d\approx\Delta t B_c$.
    """)
    return


@app.cell
def _(ca):
    def jacobians(s, u, expression):
        A = ...  # TODO: ca.jacobian with respect to the state.
        B = ...  # TODO: ca.jacobian with respect to the input.
        return ca.Function("A", [s, u], [A]), ca.Function("B", [s, u], [B])

    return (jacobians,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The continuous-time Jacobians are CasADi `Function` objects with the displayed input/output dimensions.
    """)
    return


@app.cell
def _(jacobians, rhs, s, u):
    # Continuous-time derivatives can be inspected independently of RK4.
    Ac, Bc = jacobians(s, u, rhs)
    print("State Jacobian function:", Ac)
    print("Input Jacobian function:", Bc)
    return Ac, Bc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Evaluating the Jacobians returns `DM` matrices; NumPy conversion gives numerical arrays, while `Sparsity` describes their symbolic structure.
    """)
    return


@app.cell
def _(Ac, Bc, np):
    Ac0 = np.asarray(Ac([0, 0], 0))
    Bc0 = np.asarray(Bc([0, 0], 0))
    print("Ac at the equilibrium:\n", Ac0)
    print("Bc at the equilibrium:\n", Bc0)
    print("Symbolic sparsity:", Ac.sparsity_out(0), Bc.sparsity_out(0))
    return Ac0, Bc0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Differentiating the `SX` transition expression gives the discrete-time Jacobian `Function` objects.
    """)
    return


@app.cell
def _(discrete_dynamics, jacobians, s, u):
    Ad, Bd = jacobians(s, u, discrete_dynamics)
    print("Discrete state Jacobian function:", Ad)
    print("Discrete input Jacobian function:", Bd)
    return Ad, Bd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The discrete-time Jacobians are shown alongside the Euler approximations $I+\Delta t A_c$ and $\Delta t B_c$.
    """)
    return


@app.cell
def _(Ac0, Ad, Bc0, Bd, dt, np):
    Ad0 = np.asarray(Ad([0, 0], 0))
    Bd0 = np.asarray(Bd([0, 0], 0))
    print("Ad at the equilibrium:\n", Ad0)
    print("Bd at the equilibrium:\n", Bd0)
    print("First-order approximation I + dt*Ac:\n", np.eye(2) + dt * Ac0)
    print("First-order approximation dt*Bc:\n", dt * Bc0)
    return Ad0, Bd0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Simulation
    Starting from $s_0=(\pi/2,0)$, simulate 200 intervals with $u_k=0$.
    Plot angle and angular velocity against time. What do you observe?

    Implement `simulate`, then press **Simulate** for the full rollout.
    Inspect the returned trajectory in its own cell before viewing the plots.
    """)
    return


@app.cell
def _(np):
    def simulate(transition, initial_state, inputs):
        states = np.empty((len(inputs) + 1, 2))
        states[0] = initial_state
        for k, action in enumerate(inputs):
            # Numerical CasADi outputs are DM matrices; store a flat NumPy state.
            next_state = ...  # TODO: evaluate the transition at states[k], action.
            states[k + 1] = np.asarray(next_state).ravel()
        return states

    return (simulate,)


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Simulate")
    run
    return (run,)


@app.cell
def _(F, dt, mo, np, run, simulate):
    mo.stop(not run.value)
    states = simulate(F, [np.pi / 2, 0], np.zeros(200))
    time = np.arange(len(states)) * dt
    print("Trajectory shape:", states.shape)
    states[:5]
    return states, time


@app.cell
def _(plt, states, time):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    axes[0].plot(time, states[:, 0]); axes[0].set_ylabel("Angle")
    axes[1].plot(time, states[:, 1]); axes[1].set_ylabel("Angular velocity")
    axes[1].set_xlabel("Time")
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus exercise — Friction
    Modify the model to account for friction in the pendulum hinge.
    **Hint:** model viscous friction by adding $-b\omega$ to the angular
    acceleration, for example with $b=0.1$.
    Run the simulation again and compare it with the first result.
    What has changed?
    """)
    return


if __name__ == "__main__":
    app.run()
