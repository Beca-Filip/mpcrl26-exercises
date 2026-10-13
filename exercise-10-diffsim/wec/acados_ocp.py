"""Economic MPC of the WEC as a parametric acados OCP for leap-c, and the wave forecast."""

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp
from leap_c.parameters import AcadosParameterManager

from .controller import design_lq
from .dynamics import DiscreteModel

U_SCALE = 1e5  # the OCP decides u / U_SCALE   [1e5 N]
J_SCALE = 1e3  # and minimises energy / J_SCALE [kJ]


def export_parametric_ocp(
    d: DiscreteModel,
    N_horizon: int,
    x_max: float = np.inf,
    u_max: float = np.inf,
    g_default=(1.0, 0.0),
    slack_l1: float = 1e6,
    slack_l2: float = 1e8,
    name: str = "wec",
) -> tuple[AcadosOcp, AcadosParameterManager, gym.spaces.Dict, dict[str, np.ndarray]]:
    """Economic MPC of the WEC as an acados OCP with the LQ value function as terminal cost.

    State z = (x, v), control u / U_SCALE, costs in J_SCALE. Two parameters are registered with
    an `AcadosParameterManager`, which returns the CasADi symbols the expressions are built from:

        g     differentiable, global -> acados p_global, sensitivities available:
              the MPC's excitation coefficient (2,).
        wave  non-differentiable -> stage-wise acados p: the wave state w_k at every stage,
              (N+1, 2), the forecast built from the current wave state.

    Returns the OCP, the manager, the parameter space of g and the default parameter dict.
    """
    manager = AcadosParameterManager(N_horizon=N_horizon)
    g_default = np.asarray(g_default, dtype=float)
    g = manager.register_parameter("g", g_default, differentiable=True)
    wave = manager.register_parameter("wave", np.zeros(2))
    param_space = gym.spaces.Dict(
        {"g": gym.spaces.Box(-2.0, 2.0, shape=(2,), dtype=np.float64)}
    )
    default_param = {"g": g_default}

    lq = design_lq(d)
    z, u = ca.SX.sym("z", 2), ca.SX.sym("u", 1)
    wt = ca.vertcat(
        g[0] * wave[0] + g[1] * wave[1], -g[1] * wave[0] + g[0] * wave[1]
    )  # G(g) w
    y = ca.vertcat(z, wt, u)
    S = np.diag([1.0, 1.0, 1.0, 1.0, U_SCALE])

    ocp = AcadosOcp()
    ocp.model.name = name
    ocp.model.x, ocp.model.u = z, u
    ocp.model.disc_dyn_expr = (
        ca.DM(d.A) @ z + ca.DM(d.Gw) @ wt + ca.DM(d.Bu * U_SCALE) @ u
    )
    ocp.cost.cost_type = ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = y.T @ ca.DM(S @ d.Wd @ S / J_SCALE) @ y
    ocp.model.cost_expr_ext_cost_e = (
        z.T @ ca.DM(lq.P / J_SCALE) @ z + 2 * z.T @ ca.DM(lq.Y / J_SCALE) @ wt
    )

    con, cost = ocp.constraints, ocp.cost
    con.x0 = np.zeros(2)  # overwritten at every solve
    ub = u_max / U_SCALE if np.isfinite(u_max) else 1e4  # leap-c needs bounds on u
    con.idxbu, con.lbu, con.ubu = np.array([0]), np.array([-ub]), np.array([ub])
    if np.isfinite(x_max):  # soft stroke limit, acados slack cost z s + 0.5 Z s^2
        con.idxbx = con.idxbx_e = np.array([0])
        con.lbx = con.lbx_e = np.array([-x_max])
        con.ubx = con.ubx_e = np.array([x_max])
        con.idxsbx = con.idxsbx_e = np.array([0])
        cost.zl = cost.zu = cost.zl_e = cost.zu_e = np.array([slack_l1 / J_SCALE])
        cost.Zl = cost.Zu = cost.Zl_e = cost.Zu_e = np.array([2 * slack_l2 / J_SCALE])

    o = ocp.solver_options
    o.N_horizon, o.tf = N_horizon, N_horizon * d.dt
    o.cost_scaling = np.ones(
        N_horizon + 1
    )  # acados would otherwise multiply stage costs by dt
    o.integrator_type = "DISCRETE"
    o.nlp_solver_type = "SQP"
    o.hessian_approx = "EXACT"
    o.regularize_method = "NO_REGULARIZE"
    o.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    o.qp_solver_cond_N = N_horizon
    o.qp_solver_ric_alg = 0
    o.with_solution_sens_wrt_params = True
    o.with_value_sens_wrt_params = True
    o.tol, o.qp_tol = 1e-8, 1e-9
    o.nlp_solver_max_iter, o.qp_solver_iter_max = 20, 200
    return ocp, manager, param_space, default_param


def phiw_powers(d: DiscreteModel, N: int) -> np.ndarray:
    """Phiw^k for k = 0..N, shape (N+1, 2, 2)."""
    return np.stack([np.linalg.matrix_power(d.Phiw, k) for k in range(N + 1)])


def wave_forecast(phi_pow: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-stage wave states w_k = Phiw^k w for a batch w of shape (B, 2): returns (B, N+1, 2)."""
    return np.einsum("kij,bj->bki", phi_pow, np.asarray(w, dtype=float))
