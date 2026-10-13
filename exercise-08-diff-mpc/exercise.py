import marimo

__generated_with = "0.25.1"
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
    # Exercise 08 — Differentiating acados with leap-c

    **Goal:** make the MPC from Exercise 07 a differentiable PyTorch layer with
    `leap-c`, and inspect its sensitivities.

    In this exercise, we show how to extend the acados formulation from the
    previous exercise step-by-step, to arrive at a leap-c controller layer that
    can be used to train parameters with PyTorch. As an example, we will use this
    formulation to solve MPC problems for varying values for the reference of the
    angle $\theta$ and compute their sensitivities.
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
    Make yourself familiar with the
    [`AcadosParameterManager`](https://github.com/leap-c/leap-c/blob/main/leap_c/parameters/manager.py#L15)
    class.

    The goal now is to make a modified version of `setup_ocp()`, in which the
    parameter manager class is used to introduce new parameters into the acados
    ocp formulation.

    Create the parameter manager and register 5 parameters, one for each field in
    the reference part of the cost. Make only the reference of the angle $\theta$
    differentiable. Assemble the parameters to create `yref`.

    Complete `setup_ocp_with_manager()` below.
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

        # TODO Exercise 1: Create the parameter manager.
        manager = ...

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
        # TODO Exercise 1: Use the parameter manager to obtain parameters 
        # and then build xref accordingly. 
        # Register one parameter per reference field and make
        # only xref1 (the angle reference) differentiable with `differentiable=True`.
        uref = ...
        xref = ...
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

        return ocp, manager

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Solving a batch of OCPs
    Familiarize yourself with
    [`AcadosDiffMpcTorch`](https://github.com/leap-c/leap-c/blob/main/leap_c/torch.py#L29)
    (see also
    [`AcadosDiffMpcCtx`](https://github.com/leap-c/leap-c/blob/main/leap_c/diff_mpc/function.py#L29)).

    Create an instance in the cell below and use it to solve a batch of
    optimization problems with varying reference of the angle.
    """)
    return


@app.cell
def _(mo, np, plot_sol_and_sens, torch):
    N_horizon = 5
    T_horizon = 0.25
    Fmax = 80.0
    delta_p = 0.005

    p_values = torch.arange(0.5*torch.pi, 1.5*torch.pi, delta_p).reshape(-1, 1)  # Shape (N, 1)
    batch_size = p_values.shape[0]
    # Enable gradient computation for this tensor
    p_values.requires_grad = True

    # TODO Exercise 2: Create the controller layer.
    controller_layer = ...

    x0 = torch.tensor([0.0, torch.pi, 0.0, 0.0]).repeat(batch_size, 1)  # Shape (N, 4)

    # TODO Exercise 2: Solve the batch of OCPs.
    params = ...
    ctx, u0, x, u, value = ...

    # Obtain the batched gradient du0/dp of shape (batch_size, 1)
    u0.sum().backward()
    du0_dp = p_values.grad
    return batch_size, ctx, du0_dp, p_values, u0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cell below masks the samples. For now it keeps all of them, you complete
    the mask in Exercise 5.
    """)
    return


@app.cell
def _(batch_size, ctx, du0_dp, np, p_values, u0):
    # TODO Exercise 5: Keep only the samples that converged (status 0).
    mask = np.ones(batch_size, dtype=bool)
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
def _(du0_dp_converged, mo, np, p_converged, p_failed, plot_sol_and_sens, u0_converged):
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

    fig = plot_sol_and_sens(x_values=p_converged, pis=[u0_converged],
                      senss=[du0_dp_converged, du0_dp_fd], pi_labels=pi_label_pairs,
                      sens_labels=sens_pi_label_pairs,
                      failed_x=p_failed,
                      )
    mo.mpl.interactive(fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Sensitivities
    If you run the notebook now, the batchwise derivatives
    $\frac{\partial u_0}{\partial \theta}$ of shape $(n_\mathrm{batch}, 1)$ are
    computed and plotted alongside the solutions. The solution is also used to
    compute and plot finite differences. Think about what the plots mean in the
    context of learning.
    **Note:** the cart pole has a very non-smooth solution map. Many problems have
    much better-behaved solution maps, and we look at such an example tomorrow.

    ## 4. Smoothing with $\tau_\mathrm{min}$
    Set the option `ocp.solver_options.tau_min = 1e-1` in `setup_ocp_with_manager()`
    above. This sets the minimum barrier parameter $\tau_\mathrm{min}$ of the
    interior-point solver. Run the notebook again. What do you observe?

    ## 5. Masking failed samples
    The field `ctx.status` from the `controller_layer` output contains the status
    of the solutions, one status for each sample in the batch. Only status $0$ means that the sample converged.
    Use the status to mask away the samples in the batch that did not converge.
    Also pass the samples in the batch that did *not* converge, to the plotting
    function as `failed_x` to clean the plots. Run the notebook again. What do you
    see? Complete the mask in the masking cell above.
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
