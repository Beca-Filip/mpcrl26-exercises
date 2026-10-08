import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import casadi as ca
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.linalg import solve_discrete_are
    from scipy.interpolate import RegularGridInterpolator
    from utils import plot_comparison, plot_slices

    return (
        RegularGridInterpolator,
        ca,
        mo,
        np,
        plot_comparison,
        plot_slices,
        plt,
        solve_discrete_are,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 03 — Value iteration and LQR

    In Exercise 02 you solved an OCP for a single initial state. Here we
    compute **policies** that return a torque for every state, once with LQR
    and once with value iteration, and compare them.

    ### Setup
    We use the pendulum from Exercise 01, $\dot\theta=\omega$,
    $\dot\omega=\sin\theta+u$, with state $s=(\theta,\omega)$ and torque $u$.
    The upright target is $(0,0)$; the hanging state is $(\pi,0)$. RK4 with
    $\Delta t=0.1$ gives the discrete-time dynamics $s_{k+1}=f(s_k,u_k)$, and
    we use the **discrete-time** stage cost

    $$
    \ell(s,u)=s^\top Qs+Ru^2,\qquad Q=\operatorname{diag}(100,0.01),\quad R=0.001.
    $$

    ## 1. LQR
    The helper `linear_model` differentiates the discretized dynamics $f$ at the
    upright equilibrium, which gives the linear model $s_{k+1}\approx As_k+Bu_k$.
    """)
    return


@app.cell
def _(ca, np):
    def transition(states, actions, dt=0.1):
        # The last axis holds (theta, omega); leading axes may hold a whole grid.
        x = np.asarray(states, dtype=float).copy()
        actions = np.broadcast_to(actions, x.shape[:-1])

        def f(s):
            return np.stack((s[..., 1], np.sin(s[..., 0]) + actions), axis=-1)

        h = dt / 5
        for _ in range(5):
            k1 = f(x)
            k2 = f(x + h * k1 / 2)
            k3 = f(x + h * k2 / 2)
            k4 = f(x + h * k3)
            x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x

    def linear_model(dt=0.1):
        s = ca.SX.sym("s", 2)
        u = ca.SX.sym("u")

        def f(x):
            return ca.vertcat(x[1], ca.sin(x[0]) + u)

        x = s
        h = dt / 5
        for _ in range(5):
            k1 = f(x)
            k2 = f(x + h * k1 / 2)
            k3 = f(x + h * k2 / 2)
            k4 = f(x + h * k3)
            x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # LQR needs Jacobians of the discrete map, not the continuous ODE.
        jac = ca.Function("jacobians", [s, u], [ca.jacobian(x, s), ca.jacobian(x, u)])
        A, B = jac([0, 0], 0)
        return np.asarray(A), np.asarray(B)

    return linear_model, transition


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As in the lecture, the LQR value function is $V^\star(s)=s^\top Xs$, where
    $X$ solves the discrete-time algebraic Riccati equation (DARE)

    $$
    X=Q+A^\top XA-A^\top XB\,(R+B^\top XB)^{-1}B^\top XA.
    $$

    Solve it with SciPy's
    [`solve_discrete_are`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html),
    then compute the gain

    $$
    K=(R+B^\top XB)^{-1}B^\top XA,\qquad \mu(s)=-Ks.
    $$

    Complete `lqr_gain`.
    """)
    return


@app.cell
def _(np, solve_discrete_are):
    def lqr_gain(A, B, Q, R):
        X = solve_discrete_are(A, B, Q, R)  # TODO: solution of the DARE with solve_discrete_are.
        K = np.linalg.solve(B.T @ X @ B + R, B.T @ X @ A)  # TODO: gain for mu(s) = -K s; use np.linalg.solve instead of an explicit inverse.
        return K, X

    return (lqr_gain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For checking the correctness we print below $K$ and $X$.

    **Excursion:** We further look at the magnitudes of the closed-loop
    eigenvalues of $A-BK$. For a stabilizing feedback, the eigenvalues of the controlled system should be smaller than $1$, then the system will converge towards $(0, 0)$.
    """)
    return


@app.cell
def _(linear_model, lqr_gain, np):
    A, B = linear_model()
    Q, R = np.diag([100, 0.01]), np.array([[0.001]])
    K, X = lqr_gain(A, B, Q, R)
    print("K =", K)
    print("X =", X)
    print("|eigenvalues of A - BK|:", np.abs(np.linalg.eigvals(A - B @ K)))
    return A, B, K, Q, R, X


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In part 3 we clip the LQR torque to $[-10,10]$. Is that clipped policy
    still the optimal LQR policy?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Value iteration
    For discretization we split $\theta\in[-\pi/2,2\pi]$ into 101 points, $\omega\in[-8,8]$
    into 51, and $u\in[-10,10]$ into 21. `make_grid` also precomputes the next
    states $f(s,u)$ for all grid states and actions; the cell after it shows
    the array shapes.
    """)
    return


@app.cell
def _(np, transition):
    def make_grid():
        grids = (np.linspace(-np.pi / 2, 2 * np.pi, 101), np.linspace(-8, 8, 51))
        states = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
        actions = np.linspace(-10, 10, 21)
        # Precompute all state/action successors once, not in every Bellman update.
        state_action_grid = np.broadcast_to(states[:, :, None, :], (*states.shape[:2], len(actions), 2))
        next_states = transition(state_action_grid, actions)
        return grids, states, actions, next_states

    return (make_grid,)


@app.cell
def _(make_grid):
    grids, states, actions, next_states = make_grid()
    print("states:", states.shape, " actions:", actions.shape, " next states:", next_states.shape)
    return actions, grids, next_states, states


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $V$ is only stored at the grid points, but the next state $f(s,u)$ usually
    lands between them. `interpolate_future` therefore estimates $V(f(s,u))$
    from the four neighbouring grid values (bilinear interpolation). Next
    states outside the grid get a very large value ($10^{12}$), so value
    iteration avoids them.

    **Note**: This is slightly different to the equations presented in the lecture but improves the performance ;)
    """)
    return


@app.cell
def _(RegularGridInterpolator):
    def interpolate_future(grids, values, next_states):
        interpolation = RegularGridInterpolator(grids, values, bounds_error=False, fill_value=1e12)
        # interpolation = RegularGridInterpolator(grids, values, bounds_error=False, fill_value=1e12, method="nearest") # for the extension exercise, you can try this instead of the default linear interpolation.
        return interpolation(next_states.reshape(-1, 2)).reshape(next_states.shape[:-1])

    return (interpolate_future,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Value iteration repeats the optimal Bellman update on all grid states until
    it converges. Our dynamics are deterministic, so the sum over next states
    $s'$ from the lecture reduces to $f(s,u)$:

    $$
    V_{\rm new}(s)=\min_u\big\{\ell(s,u)+\gamma\,V\big(f(s,u)\big)\big\},
    \qquad
    \Delta=\max_s\big|V_{\rm new}(s)-V(s)\big|.
    $$

    Set $V\gets V_{\rm new}$ and repeat until $\Delta<\varepsilon$; for
    $\gamma<1$, $V$ converges to $V^\star$. Finally, act greedily with respect
    to $V$:

    $$
    \mu(s)\in\operatorname*{arg\,min}_u\big\{\ell(s,u)+\gamma\,V\big(f(s,u)\big)\big\}.
    $$

    Complete `value_iteration`. It starts from $V=0$ and returns the converged
    values and the greedy policy, both with shape `(101, 51)`, and the number
    of iterations.
    """)
    return


@app.cell
def _(interpolate_future, np):
    def value_iteration(grids, states, actions, next_states, Q, R, gamma, tolerance=1e-2):
        stage = np.einsum("...i,ij,...j->...", states, Q, states)[..., None] + R * actions**2 # TODO: stage costs l(s, u) with shape (101, 51, 21): s^T Q s per state plus R u^2 per action.
        V = np.zeros(states.shape[:-1])
        for iteration in range(1, 10_000):
            future = interpolate_future(grids, V, next_states)
            costs = stage + gamma * future  # TODO: l(s, u) + gamma V(f(s, u)) for every state and action.
            V_new = np.min(costs, axis=-1)  # TODO: minimize over the action axis.
            delta = np.max(np.abs(V_new - V), axis=None)  # TODO: largest change max_s |V_new(s) - V(s)|.
            V = V_new
            if delta < tolerance:
                break
        # Greedy policy with respect to the converged V.
        costs = stage + gamma * interpolate_future(grids, V, next_states)
        policy = actions[costs.argmin(axis=-1)]
        return V, policy, iteration

    return (value_iteration,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use $\gamma=0.99$ and $\varepsilon=10^{-2}$. The cell below prints the
    number of iterations and $V$ at the upright and at the hanging state.
    """)
    return


@app.cell
def _(
    Q,
    R,
    RegularGridInterpolator,
    actions,
    grids,
    next_states,
    np,
    states,
    value_iteration,
):
    gamma = 0.99
    V, policy, iterations = value_iteration(grids, states, actions, next_states, Q, R.item(), gamma)
    V_interpolated = RegularGridInterpolator(grids, V)
    print("Converged after", iterations, "iterations")
    print("V at upright (0, 0):", V_interpolated([[0, 0]])[0])
    print("V at hanging (pi, 0):", V_interpolated([[np.pi, 0]])[0])
    return V, gamma, policy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Why does value iteration need $\gamma<1$, while the LQR uses no discount?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Compare LQR and value iteration
    The plots compare the cost-to-go and the policies on the grid. The LQR
    torque is clipped to $[-10,10]$.
    """)
    return


@app.cell
def _(K, V, X, grids, np, plot_comparison, policy, states):
    lqr_value = np.einsum("...i,ij,...j->...", states, X, states)
    lqr_action = np.clip(-(states @ K.T)[..., 0], -10, 10)
    plot_comparison(grids, V, policy, lqr_value, lqr_action)
    return (lqr_value,)


@app.cell
def _(V, grids, lqr_value, plot_slices):
    plot_slices(grids, V, lqr_value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Where do the cost-to-go and the policies agree? Why does LQR lose accuracy
    far from the upright equilibrium?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cell below simulates both policies in closed loop for 10 s, from a
    small perturbation $(0.1,0)$ and from the hanging state $(\pi,0)$. The
    value-iteration policy acts greedily with respect to $V$ at the exact
    state, not only at grid points.
    """)
    return


@app.cell
def _(
    K,
    Q,
    R,
    V,
    actions,
    gamma,
    grids,
    interpolate_future,
    mo,
    np,
    plt,
    transition,
):
    def vi_policy(state):
        # Greedy action at the exact state: evaluate l(s, u) + gamma V(f(s, u)) for all actions.
        successors = transition(np.broadcast_to(state, (len(actions), 2)), actions)
        costs = state @ Q @ state + R.item() * actions**2 + gamma * interpolate_future(grids, V, successors)
        return actions[costs.argmin()]

    def lqr_policy(state):
        return np.clip(-(K @ state).item(), -10, 10)

    def simulate(policy_function, initial_state, steps=100):
        trajectory = [np.asarray(initial_state, dtype=float)]
        for _ in range(steps):
            trajectory.append(transition(trajectory[-1], policy_function(trajectory[-1])))
        return np.asarray(trajectory)

    _figures = []
    for _initial in [(0.1, 0.0), (np.pi, 0.0)]:
        _fig, _axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
        for _name, _policy in [("Value iteration", vi_policy), ("LQR", lqr_policy)]:
            _trajectory = simulate(_policy, _initial)
            _time = 0.1 * np.arange(len(_trajectory))
            _axes[0].plot(_time, _trajectory[:, 0], label=_name)
            _axes[1].plot(_time, _trajectory[:, 1])
        _axes[0].set(title=f"Closed loop from ({_initial[0]:.2f}, 0)", ylabel="Angle")
        _axes[0].legend()
        _axes[1].set(xlabel="Time", ylabel="Angular velocity")
        _fig.tight_layout()
        _figures.append(_fig)
    mo.vstack(_figures)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Which policy brings the pendulum up from $(\pi,0)$ faster? Why does value
    iteration keep a small oscillation around upright?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Extra: Riccati iteration
    `solve_discrete_are` hides how $X$ is computed. As in the lecture, iterate
    the Riccati recursion from $X=Q$ until the largest change is below a
    tolerance:

    $$
    X_{\rm new}=Q+A^\top XA-A^\top XB\,(R+B^\top XB)^{-1}B^\top XA.
    $$

    This is value iteration for LQR: the value function $s^\top Xs$ is stored
    as a matrix instead of on a grid. The cell compares the result with SciPy's $X$.
    """)
    return


@app.cell
def _(A, B, Q, R, X, np):
    def riccati_iteration(A, B, Q, R, tolerance=1e-9):
        X_k = Q
        for iteration in range(1, 10_000):
            X_next = Q + A.T @ X_k @ A - A.T @ X_k @ B @ np.linalg.solve(R + B.T @ X_k @ B, B.T @ X_k @ A)  # TODO: one step of the Riccati recursion.
            delta = np.abs(X_next - X_k).max()
            X_k = X_next
            if delta < tolerance:
                break
        return X_k, iteration

    X_iterated, riccati_iterations = riccati_iteration(A, B, Q, R)
    print("Converged after", riccati_iterations, "iterations")
    print("max |X_iterated - X|:", np.abs(X_iterated - X).max())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Extensions:** vary $\gamma$, or refine the state or action grid. Note that
    the angle is not wrapped: $(2\pi,0)$ is upright, but counts as far from the
    target.

    Finally, switch off the interpolation: pass `method="nearest"` to
    `RegularGridInterpolator` in `interpolate_future`, so that $V$ is taken from
    the nearest grid point. What changes in the closed loop near upright?
    """)
    return


if __name__ == "__main__":
    app.run()
