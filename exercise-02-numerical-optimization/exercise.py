import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import casadi as ca
    import numpy as np
    from utils import plot_nlp, plot_ocp

    return ca, mo, np, plot_nlp, plot_ocp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 02 — Numerical optimization and optimal control

    **Goal:** use CasADi to formulate, differentiate, and solve constrained
    optimization problems, then apply these tools to pendulum swing-up.

    ## 1. Derivatives and KKT conditions
    The following problem combines a nonlinear objective with a curved feasible set:

    $$
    \begin{aligned}
    \min_{x,y}\quad &f(x,y)=\tfrac12(x-1)^2+50(y-x^2)^2+\tfrac12x^2,\\
    \text{s.t.}\quad &g(x,y)=x+(1-y)^2=0.
    \end{aligned}
    $$

    **Task — Derivatives:** derive the gradients and Hessians of $f$ and $g$ on paper.

    **Task — KKT conditions:** write out the KKT equations for $L=f+\lambda g$;
    are these conditions necessary for optimality, and are they sufficient?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now move from the pen-and-paper calculations to CasADi code.

    **Task:** define the objective `f` and constraint `g` as `SX` expressions;
    see CasADi's [SX symbolics documentation](https://web.casadi.org/docs/#the-sx-symbolics).
    """)
    return


@app.cell
def _(ca):
    z = ca.SX.sym("z", 2)
    x, y = z[0], z[1]
    f = 1/2 * (x - 1)**2 + 50 * (y - x**2)**2 + 1/2 * x**2  # TODO: scalar objective expression.
    g = x + (1 - y)**2  # TODO: scalar equality-constraint expression.
    print("Objective expression:", f)
    print("Constraint expression:", g)
    return f, g, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Task:** wrap `f` and `g` in CasADi `Function` objects for numerical evaluation.
    """)
    return


@app.cell
def _(ca, f, g, z):
    objective = ca.Function("objective", [z], [f])  # TODO: Function with input z and output f.
    constraint = ca.Function("constraint", [z], [g])  # TODO: Function with input z and output g.
    print(objective)
    print(constraint)
    return constraint, objective


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Evaluating the objective and constraint at `initial_guess` returns CasADi `DM` objects.
    """)
    return


@app.cell
def _(constraint, np, objective):
    initial_guess = np.array([0.0, 0.0])
    print("f(z):", objective(initial_guess))
    print("g(z):", constraint(initial_guess))
    return (initial_guess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Task:** generate the gradient and Hessian expressions with CasADi; `derivatives` collects them as named outputs.
    See CasADi's [differentiation documentation](https://web.casadi.org/docs/#calculus-algorithmic-differentiation) for `gradient` and `hessian`.
    """)
    return


@app.cell
def _(ca, f, g, z):
    grad_f = ca.gradient(f, z)  # TODO: gradient of f with respect to z.
    grad_g = ca.gradient(g, z)  # TODO: gradient of g with respect to z.
    hess_f = ca.hessian(f, z)[0]  # TODO: Hessian of f; ca.hessian returns (Hessian, gradient).
    hess_g = ca.hessian(g, z)[0]  # TODO: Hessian of g.
    derivatives = ca.Function("derivatives", [z], [grad_f, grad_g, hess_f, hess_g],
                              ["z"], ["grad_f", "grad_g", "hess_f", "hess_g"])
    derivatives
    return (derivatives,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The gradients and Hessians at `initial_guess` are returned as `DM` matrices.
    """)
    return


@app.cell
def _(derivatives, initial_guess):
    # Named outputs make it clear which derivative each matrix represents.
    derivative_values = derivatives(z=initial_guess)
    for _name, _value in derivative_values.items():
        print(f"{_name}:\n{_value}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `nlpsol` creates an IPOPT solver represented by a CasADi `Function`.
    **Task:** construct `solver` for the problem dictionary `nlp` using `ca.nlpsol`.
    """)
    return


@app.cell
def _(ca, f, g, z):
    nlp = {"x": z, "f": f, "g": g}
    solver = ca.nlpsol("solver", "ipopt", nlp)  # TODO: create an IPOPT solver for nlp.
    solver
    return (solver,)


@app.cell
def _(mo):
    solve_small = mo.ui.run_button(label="Solve the constrained NLP")
    solve_small
    return (solve_small,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calling the solver with an initial guess and equality bounds returns a dictionary of `DM` results.
    **Task:** set `lbg` and `ubg` to enforce the equality constraint.
    """)
    return


@app.cell
def _(initial_guess, mo, solve_small, solver):
    mo.stop(not solve_small.value)
    # Equal lower and upper constraint bounds enforce g(z) = 0.
    result = solver(x0=initial_guess, lbg=0., ubg=0.)  # TODO: equality bounds.
    print("Result keys:", list(result))
    print("Solver status:", solver.stats()["return_status"])
    result
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The solver result contains the optimal variables, objective value, and equality-constraint multiplier.
    **Task:** extract `z_star` and `lambda_star` from the result dictionary.
    """)
    return


@app.cell
def _(result):
    z_star = result["x"]  # TODO: optimal primal variables.
    lambda_star = result["lam_g"]  # TODO: equality-constraint multiplier.
    print("Primal solution:", z_star)
    print("Objective value:", result["f"])
    print("Constraint multiplier:", lambda_star)
    return lambda_star, z_star


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Task:** complete `stationarity` using the symbolic multiplier `lam` to obtain the KKT residual function.
    """)
    return


@app.cell
def _(ca, f, g, z):
    lam = ca.SX.sym("lam")
    stationarity = f + lam * g  # TODO: gradient of the Lagrangian f + lam*g.
    kkt = ca.Function("kkt", [z, lam], [g, stationarity],
                      ["z", "lam"], ["feasibility", "stationarity"])
    kkt
    return (kkt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The KKT residuals evaluated at the primal and dual solution should both be close to zero.
    """)
    return


@app.cell
def _(kkt, lambda_star, z_star):
    residuals = kkt(z=z_star, lam=lambda_star)
    print("Feasibility residual:", residuals["feasibility"])
    print("Stationarity residual:\n", residuals["stationarity"])
    return


@app.cell
def _(plot_nlp, z_star):
    plot_nlp(z_star)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Pendulum swing-up by multiple shooting
    Use Exercise 01's dynamics to move the pendulum from its hanging state
    $(-\pi,0)$ toward the upright target $(0,0)$ with bounded torque.

    $$
    \begin{aligned}
    \min_{s_0,u_0,\ldots,s_N}\quad &
    \frac{\Delta t}{2}\sum_{k=0}^{N-1}(s_k^TQs_k+u_k^2)+\frac12s_N^TQs_N\\
    \text{s.t.}\quad&s_0=(-\pi,0),\quad s_{k+1}=F(s_k,u_k),\\
    &-1\leq u_k\leq1,\quad -\pi\leq\theta_k\leq2\pi.
    \end{aligned}
    $$

    Use $N=200$, $\Delta t=0.05$, $Q=\operatorname{diag}(10,1)$; angular
    velocity is unconstrained. The lower angle bound $-\pi$ keeps the hanging
    initial state admissible.
    Introduce state and input variables, enforce dynamics with shooting
    equalities, and add stage and terminal costs before solving with IPOPT.

    See CasADi's [Opti documentation](https://web.casadi.org/docs/#opti-stack).
    """)
    return


@app.cell
def _(ca):
    def pendulum_step(state, action, dt):
        def f(x):
            return ca.vertcat(x[1], ca.sin(x[0]) + action)

        # Reuse Exercise 01's RK4 scheme with five substeps per control interval.
        h = dt / 5
        for _ in range(5):
            k1 = f(state)
            k2 = f(state + h * k1 / 2)
            k3 = f(state + h * k2 / 2)
            k4 = f(state + h * k3)
            state = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return state

    return (pendulum_step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The RK4 transition is wrapped in a CasADi `Function` and evaluated for one sampling interval.
    """)
    return


@app.cell
def _(ca, pendulum_step):
    dt = 0.05
    pendulum_state = ca.SX.sym("s", 2)
    pendulum_input = ca.SX.sym("u")
    F = ca.Function("F", [pendulum_state, pendulum_input],
                    [pendulum_step(pendulum_state, pendulum_input, dt)])
    F
    return F, dt


@app.cell
def _(F, np):
    step_probe = F([-np.pi + 0.1, 0], 0)
    print("State after one interval:", step_probe)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Opti.variable` creates `MX` state and input variables of shapes `(2,N+1)` and `(1,N)`.
    """)
    return


@app.cell
def _(ca):
    N = 200
    Q = ca.diag(ca.DM([10.0, 1.0]))
    ocp_variables = ca.Opti()
    X = ocp_variables.variable(2, N + 1)
    U = ocp_variables.variable(1, N)
    print("Variable types:", type(X), type(U))
    print("State shape:", X.shape, "input shape:", U.shape)
    return N, Q, U, X, ocp_variables


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Task:** set `initial_state` for the initial-state equality; an `Opti` copy keeps the preceding stage unchanged.
    """)
    return


@app.cell
def _(U, X, ca, np, ocp_variables):
    ocp_bounds = ocp_variables.copy()
    initial_state = ca.DM([0, 1])  # TODO: use a CasADi DM for (-pi, 0).
    ocp_bounds.subject_to(X[:, 0] == initial_state)
    ocp_bounds.subject_to(ocp_bounds.bounded(-1, U, 1))
    ocp_bounds.subject_to(ocp_bounds.bounded(-np.pi, X[0, :], 2 * np.pi))
    print("Constraint-vector shape:", ocp_bounds.g.shape)
    return (ocp_bounds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Task:** complete `_gap` so the shooting equalities link the independent `MX` states through the transition function.
    """)
    return


@app.cell
def _(F, N, U, X, ocp_bounds):
    ocp_dynamics = ocp_bounds.copy()
    shooting_gaps = []
    for _k in range(N):
        _next_state = F(X[:, _k], U[0, _k])
        _gap = X[:, _k + 1] - _next_state  # TODO: X[:, _k + 1] minus the integrated next state.
        ocp_dynamics.subject_to(_gap == 0)
        shooting_gaps.append(_gap)
    print("Constraint-vector shape with dynamics:", ocp_dynamics.g.shape)
    return ocp_dynamics, shooting_gaps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Task:** complete `_stage_cost` and `terminal_cost` to define the scalar `MX` objective `Opti.f`.
    """)
    return


@app.cell
def _(N, Q, X, dt, ocp_dynamics):
    ocp = ocp_dynamics.copy()
    cost = 0
    for _k in range(N):
        _stage_cost = dt / 2 * (X[:, _k].T @ Q @ X[:, _k])  # TODO: dt/2 times the state and input quadratic penalties.
        cost += _stage_cost
    terminal_cost = 1/2 * X[:, -1].T @ Q @ X[:, -1]  # TODO: endpoint penalty, without a dt factor.
    ocp.minimize(cost + terminal_cost)
    print("Objective type:", type(ocp.f), "shape:", ocp.f.shape)
    return (ocp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Opti.set_initial` sets the initial guess, and `Opti.solver` configures IPOPT without solving the problem.
    """)
    return


@app.cell
def _(N, U, X, np, ocp):
    ocp_ready = ocp.copy()
    # An initial guess need not satisfy the shooting constraints.
    ocp_ready.set_initial(X[0, :], np.linspace(-np.pi, 0, N + 1))
    ocp_ready.set_initial(X[1, :], 0)
    ocp_ready.set_initial(U, 0)
    ocp_ready.solver("ipopt", {"print_time": False}, {"print_level": 0, "max_iter": 2000})
    return (ocp_ready,)


@app.cell
def _(mo):
    solve_control = mo.ui.run_button(label="Solve the swing-up OCP")
    solve_control
    return (solve_control,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Solving the OCP returns an `OptiSol` object for evaluating expressions at the optimized variables.
    """)
    return


@app.cell
def _(mo, ocp_ready, solve_control):
    mo.stop(not solve_control.value)
    control_solution = ocp_ready.solve()
    print("Solution type:", type(control_solution))
    return (control_solution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `OptiSol.value` extracts numerical trajectories and shooting gaps; the largest absolute gap measures dynamics feasibility.
    """)
    return


@app.cell
def _(U, X, ca, control_solution, np, shooting_gaps):
    states = np.asarray(control_solution.value(X)).T
    actions = np.asarray(control_solution.value(U)).ravel()
    gap_values = np.asarray(control_solution.value(ca.vertcat(*shooting_gaps)))
    print("State trajectory:", states.shape, "input trajectory:", actions.shape)
    print("Maximum shooting gap:", np.max(np.abs(gap_values)))
    states[:5]
    return actions, states


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Forward simulation of the optimized inputs provides an independent check against the shooting states.
    """)
    return


@app.cell
def _(F, actions, np, states):
    simulated_states = np.empty_like(states)
    simulated_states[0] = states[0]
    for _k, _action in enumerate(actions):
        simulated_states[_k + 1] = np.asarray(F(simulated_states[_k], _action)).ravel()
    print("Maximum state difference:", np.max(np.abs(simulated_states - states)))
    return


@app.cell
def _(actions, dt, plot_ocp, states):
    plot_ocp(states, actions, dt)
    return


if __name__ == "__main__":
    app.run()
