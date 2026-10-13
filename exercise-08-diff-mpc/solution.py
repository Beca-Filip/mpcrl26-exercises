import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import casadi as ca
    import numpy as np
    import torch
    from acados_template import AcadosOcp
    from leap_c.parameters import AcadosParameterManager
    from leap_c.torch import AcadosDiffMpcTorch
    from pendulum_model import export_pendulum_model
    from utils import plot_sol_and_sens

    return (
        AcadosDiffMpcTorch,
        AcadosOcp,
        AcadosParameterManager,
        ca,
        export_pendulum_model,
        mo,
        np,
        plot_sol_and_sens,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 08 — Differentiating acados with leap-c (Solution)

    **Goal:** make the MPC from Exercise 07 a differentiable PyTorch layer with
    `leap-c`, and inspect its sensitivities.

    The OCP is extended with an `AcadosParameterManager`, which introduces the
    reference fields as differentiable parameters. `AcadosDiffMpcTorch` then solves
    a batch of OCPs and backpropagates through the solver, so the sensitivity
    $\frac{\partial u_0}{\partial \theta_\mathrm{ref}}$ is available in PyTorch.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <p align="center">
      <img src="public/cartpole.png" width="420" alt="An illustration of Cart Pole while swinging up."><br>
      <em>An illustration of Cart Pole while swinging up.</em>
    </p>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Parameters in the OCP
    The [`AcadosParameterManager`](https://github.com/leap-c/leap-c/blob/main/leap_c/parameters/manager.py#L15) collects named parameters and assigns each a
    default. Parameters registered with `differentiable=True` are inputs for which
    the solver provides derivatives.

    The parameter manager is created first, then one parameter is registered per
    reference field. Only `xref1` (the angle reference $\theta$) is differentiable.
    """)
    return


@app.cell
def _(AcadosOcp, AcadosParameterManager, ca, export_pendulum_model, np):
    def setup_ocp_with_manager(Fmax: float, N_horizon: int, Tf: float) -> tuple[AcadosOcp, AcadosParameterManager]:
        """Create an ocp object to formulate the OCP for the cartpole system.

        Args:
            Fmax: Maximum force that can be applied to the cart.
            N_horizon: Number of steps in the prediction horizon.
            Tf: Total time horizon of the prediction horizon.
            param_manager: Parameter manager that holds the parameters to be used in the ocp.
        """
        # create ocp object to formulate the OCP
        ocp = AcadosOcp()

        # set model
        ocp.model = export_pendulum_model(dt=Tf/N_horizon)
        ocp.solver_options.integrator_type = "DISCRETE"

        nx = ocp.model.x.rows()

        # Create the parameter manager.
        manager = AcadosParameterManager(N_horizon=N_horizon)

        # ======================= set cost =======================
        # NOTE: NONLINEAR_LS pseudo: 0.5 * (y_expr - yref)^T W (y_expr - yref)
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        # weight matrices
        q_diag_sqrt = np.sqrt(np.array([1e3, 1e3, 1e-2, 1e-2]))
        r_diag_sqrt = np.sqrt(np.array([1e-2]))
        W_sqrt = ca.diag(ca.vertcat(q_diag_sqrt, r_diag_sqrt))
        ocp.cost.W = W_sqrt @ W_sqrt.T
        ocp.cost.W_e = ocp.cost.W[: nx, : nx]

        # expressions for ys
        ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)
        ocp.model.cost_y_expr_e = ocp.model.x

        # references
        # Use the parameter manager to obtain parameters and then build xref accordingly.
        xref0 = manager.register_parameter(
            "xref0",
            default=np.array([0.0]),
        )
        xref1 = manager.register_parameter(
            "xref1",
            default=np.array([0.0]),
            differentiable=True,
        )
        xref2 = manager.register_parameter(
            "xref2",
            default=np.array([0.0]),
        )
        xref3 = manager.register_parameter(
            "xref3",
            default=np.array([0.0]),
        )
        uref = manager.register_parameter(
            "uref",
            default=np.array([0.0]),
        )
        xref = ca.vertcat(xref0, xref1, xref2, xref3)
        yref = ca.vertcat(xref, uref)
        yref_e = yref[: nx]
        ocp.cost.yref = yref
        ocp.cost.yref_e = yref_e

        # ============== set constraints ================
        # Actuator constraints
        ocp.constraints.lbu = np.array([-Fmax])
        ocp.constraints.ubu = np.array([+Fmax])
        ocp.constraints.idxbu = np.array([0])

        # Will be changed in the closed loop
        ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

        # ============== set ocp options ================
        # prediction horizon
        ocp.solver_options.N_horizon = N_horizon
        ocp.solver_options.tf = Tf

        # additional ocp options
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.qp_tol = 1e-4
        ocp.solver_options.tol = 1e-3
        ocp.solver_options.nlp_solver_max_iter = 1000

        # Exercise 4: uncomment to smooth the solution map (see section 4).
        # ocp.solver_options.tau_min = 1e-1

        return ocp, manager

    return (setup_ocp_with_manager,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Solving a batch of OCPs
    [`AcadosDiffMpcTorch`](https://github.com/leap-c/leap-c/blob/main/leap_c/torch.py#L29) wraps the OCP and the parameter manager into a
    `torch.nn.Module` that solves the OCP for a batch of initial states and
    parameters and exposes the solver context ([`AcadosDiffMpcCtx`](https://github.com/leap-c/leap-c/blob/main/leap_c/diff_mpc/function.py#L29)).

    The controller layer is created once, then
    called with the batched initial state `x0` and the batched parameters `params`.

    ## 3. Sensitivities
    Calling `u0.sum().backward()` differentiates through the MPC solution, filling `p_values.grad` with the batchwise
    derivatives $\frac{\partial u_0}{\partial \theta}$ of shape
    $(n_\mathrm{batch}, 1)$. By the way, computing the jacobian here with .backward() only works this way because
    1. The samples in a batch are independend of each other
    2. $u_0$ has scalar dimension

    The solution is compared with finite differences.
    Where the solver is smooth and converged, the two agree closely; the sensitivity
    is exactly the gradient a learning algorithm would use to train the reference.
    However, the solution gradient is 0 over a large portion of the parameter space, and it jumps heavily at some
    points.
    **Note:** the cart pole has a very non-smooth solution map. Many problems have
    much better-behaved solution maps, and we look at such an example tomorrow.
    """)
    return


@app.cell
def _(AcadosDiffMpcTorch, np, setup_ocp_with_manager, torch):
    N_horizon = 5
    T_horizon = 0.25
    Fmax = 80.0
    delta_p = 0.005

    p_values = torch.arange(0.5*torch.pi, 1.5*torch.pi, delta_p).reshape(-1, 1)  # Shape (N, 1)
    batch_size = p_values.shape[0]
    # Enable gradient computation for this tensor
    p_values.requires_grad = True

    # Create the controller layer.
    ocp, param_manager = setup_ocp_with_manager(Fmax=Fmax,
                                            N_horizon=N_horizon,
                                            Tf=T_horizon)
    controller_layer = AcadosDiffMpcTorch(
        ocp=ocp,
        parameter_manager=param_manager,
    )

    x0 = torch.tensor([0.0, torch.pi, 0.0, 0.0]).repeat(batch_size, 1)  # Shape (N, 4)

    # Solve the batch of OCPs.
    params = {"xref1": p_values}
    ctx, u0, x, u, value = controller_layer(x0=x0, u0=None, params=params, ctx=None)

    # Obtain the batched gradient du0/dp of shape (batch_size, 1)
    u0.sum().backward()
    du0_dp = p_values.grad
    return batch_size, ctx, du0_dp, p_values, u0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Only samples with `ctx.status == 0` converged. The others are masked out
    (Exercise 5).
    """)
    return


@app.cell
def _(batch_size, ctx, du0_dp, p_values, u0):
    mask = ctx.status == 0
    n_converged = mask.sum()
    assert n_converged >= 2, "Need at least two converged samples to plot."
    print(f"{n_converged} samples out of {batch_size} converged.")
    # Convert to NumPy for plotting.
    p_converged = p_values.detach().numpy()[mask]
    u0_converged = u0.detach().numpy().flatten()[mask]
    du0_dp_converged = du0_dp.numpy().flatten()[mask]
    p_failed = p_values.detach().numpy()[~mask]
    return du0_dp_converged, p_converged, p_failed, u0_converged


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The left plot shows the first action $u_0$ over the angle reference
    $\theta_\mathrm{ref}$. The right plot shows its sensitivity from acados next to
    finite differences of $u_0$. Red dots mark samples that did not converge.
    """)
    return


@app.cell
def _(
    du0_dp_converged,
    np,
    p_converged,
    p_failed,
    plot_sol_and_sens,
    u0_converged,
):
    du0_dp_fd = np.gradient(u0_converged, p_converged.flatten())

    test_tol = 1e-3
    median_diff = np.median(np.abs(du0_dp_converged - du0_dp_fd))
    print("Median difference between policy gradient obtained by acados and via FD is "
          f"{median_diff} should be < {test_tol}.")

    # solutions to plot
    label = r'solver'
    pi_label_pairs = []
    sens_pi_label_pairs = []

    pi_label_pairs.append(label)
    sens_pi_label_pairs.append(label)

    sens_pi_label_pairs.append('finite diff.')

    plot_sol_and_sens(x_values=p_converged, pis=[u0_converged],
                      senss=[du0_dp_converged, du0_dp_fd], pi_labels=pi_label_pairs,
                      sens_labels=sens_pi_label_pairs,
                      failed_x=p_failed,
                      )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Smoothing with $\tau_\mathrm{min}$
    The minimum barrier parameter $\tau_\mathrm{min}$ (solver option `tau_min`) of the
    interior-point solver bounds how far the barrier is reduced.
    Increasing it from its default to $10^{-1}$ makes the solution less accurate, but smoother. However, now a part of the samples
    do not converge anymore, drastically changing the plots.

    To see this, uncomment `ocp.solver_options.tau_min = 1e-1` in
    `setup_ocp_with_manager()` above and rerun the notebook.

    ## 5. Masking failed samples
    When differentiating through NMPC, it is hard to guarantee
    that the solver converges for all sorts of different parameters and settings.
    A solution can be to just remove the samples that did not converge.
    `ctx.status == 0` selects the samples whose solver attempt converged. Masking
    them out removes the solver failures from the sensitivity plot and passes the
    failed reference values as `failed_x`, so the plots only show meaningful samples.
    Now we can see the smoothed plots. Note that while the actual solution does not seem to differ too much,
    the gradient is nonzero (albeit still close to it in a region) everywhere and the kinks are less sharp. Therefore, even for samples in the flat (or "dead") region, some gradient signal is still backpropagated in a learning setting.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Bonus and resources
    Feel free to play around with, e.g., values of the parameters, ocp
    formulations, solver options, calculating other sensitivities, making
    parameters differ over the horizon (field `splits` in `register_parameter`), ...

    **Useful resources:**
    - [acados problem formulation](https://github.com/acados/acados/blob/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf): the general OCP that acados solves, with all cost and constraint types
    - [leap-c-lab](https://github.com/leap-c/leap-c-lab): example environments, OCPs and planners built on leap-c
    """)
    return


if __name__ == "__main__":
    app.run()
