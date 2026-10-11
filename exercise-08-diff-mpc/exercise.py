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

    In this exercise, we show how to extend the acados formulation from the
    previous exercise step-by-step, to arrive at a leap-c controller layer that
    can be used to train parameters with PyTorch. As an example, we will use this
    formulation to solve MPC problems for varying values for the reference of the
    angle $\theta$ and compute their sensitivities.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("img/cartpole.png", width=420,
             caption="An illustration of Cart Pole while swinging up.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Parametric sensitivities

    ### 1.1 The parameter manager
    Make yourself familiar with the
    [`AcadosParameterManager`](https://github.com/leap-c/leap-c/blob/main/leap_c/parameters/manager.py#15)
    class.

    ### 1.2 Introducing parameters into the OCP
    The goal now is to make a modified version of `setup_ocp()`, in which the
    parameter manager class is used to introduce new parameters into the acados
    ocp formulation.

    Create the parameter manager and register 5 parameters, one for each field in
    the reference part of the cost. Make only the reference of the angle $\theta$
    learnable. Assemble the parameters to create `yref`.

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

        # TODO Exercise 1.2: Create the parameter manager.
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
        # TODO Exercise 1.2: Use the parameter manager to obtain parameters 
        # and then build xref accordingly. 
        # Register one parameter per reference field and make
        # only xref1 (the angle reference) learnable via differentiable=True.
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

        # ============= Finalize =============
        ocp.code_export_directory = 'c_generated_code_ocp'

        return ocp, manager

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.3 The differentiable MPC layer
    Familiarize yourself with
    [`AcadosDiffMpcTorch`](https://github.com/leap-c/leap-c/blob/main/leap_c/torch.py#L29)
    (see also
    [`AcadosDiffMpcCtx`](https://github.com/leap-c/leap-c/blob/main/leap_c/diff_mpc/function.py#L29)).

    ### 1.4 Solving a batch of OCPs
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

    p_values_npy = np.arange(0.5*np.pi, 1.5*np.pi, delta_p)
    p_values_npy = p_values_npy.reshape(-1, 1)  # Shape (N, 1)
    batch_size = p_values_npy.shape[0]
    p_values = torch.tensor(p_values_npy, dtype=torch.float32)
    # Enable gradient computation for this tensor
    p_values.requires_grad = True

    # TODO Exercise 1.4: Create the controller layer.
    controller_layer = ...

    x0_npy = np.array([0.0, np.pi, 0.0, 0.0])
    x0_npy = np.tile(x0_npy, (p_values_npy.shape[0], 1))  # Shape (N, 4)
    x0 = torch.tensor(x0_npy, dtype=torch.float32)

    # TODO Exercise 1.4: Solve the batch of OCPs.
    params = ...
    ctx, u0, x, u, value = ...

    # Obtain the batched gradient du0/dp of shape (batch_size, 1)
    u0.sum().backward()
    du0_dp = p_values.grad

    du0_dp_npy = du0_dp.detach().numpy().flatten()
    u0_npy = u0.detach().numpy().flatten()

    # TODO Exercise 1.7: Mask here.
    # mask = ...
    # u0_npy = ...
    # du0_dp_npy = ...
    # p_values_npy_failed = ...
    # p_values_npy = ...

    du0_dp_fd = np.gradient(u0_npy, p_values_npy.flatten())

    test_tol = 1e-3
    median_diff = np.median(np.abs(du0_dp_npy - du0_dp_fd))
    print("Median difference between policy gradient obtained by acados and via FD is "
          f"{median_diff} should be < {test_tol}.")

    # solutions to plot
    label = r'solver'
    pi_label_pairs = []
    sens_pi_label_pairs = []

    pi_label_pairs.append(label)
    sens_pi_label_pairs.append(label)

    sens_pi_label_pairs.append('finite diff.')

    fig = plot_sol_and_sens(x_values=p_values_npy, pis=[u0_npy],
                      senss=[du0_dp_npy, du0_dp_fd], pi_labels=pi_label_pairs,
                      sens_labels=sens_pi_label_pairs,
                      )
    mo.mpl.interactive(fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.5 Sensitivities
    If you run the notebook now, the batchwise derivatives
    $\frac{\partial u_0}{\partial \theta}$ of shape $(n_\mathrm{batch}, 1)$ are
    computed and plotted alongside the solutions. The solution is also used to
    compute and plot finite differences. Think about what the plots mean in the
    context of learning.

    ### 1.6 The `tau_min` solver option
    Set the option `ocp.solver_options.tau_min = 1e-1` in `setup_ocp_with_manager()`
    above. This basically sets the minimum barrier parameter
    inside the interior point solving. Run the notebook again. What do you observe?

    ### 1.7 Masking samples
    The field `ctx.status` from the `controller_layer` output contains the status
    of the solutions, one status for each sample in the batch. Only status $0$ means that the sample converged.
    Use the status to mask away the samples in the batch that did not converge.
    Also pass the samples in the batch that did *not* converge, to the plotting
    function as `failed_x` to clean the plots. Run the notebook again. What do you
    see? Complete the mask in the cell below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.8 Bonus
    Feel free to play around with, e.g., values of the parameters, ocp
    formulations, solver options, calculating other sensitivities, making
    parameters differ over the horizon (field `splits` in `register_parameter`), ...

    ### Useful resources
    - [acados problem formulation PDF](https://github.com/acados/acados/blob/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf)
    - [`AcadosParameterManager`](https://github.com/leap-c/leap-c/blob/main/leap_c/parameters/manager.py#15)
    - [`AcadosDiffMpcTorch`](https://github.com/leap-c/leap-c/blob/main/leap_c/torch.py#L29)
    - [`AcadosDiffMpcCtx`](https://github.com/leap-c/leap-c/blob/main/leap_c/diff_mpc/function.py#L29)
    """)
    return


if __name__ == "__main__":
    app.run()
