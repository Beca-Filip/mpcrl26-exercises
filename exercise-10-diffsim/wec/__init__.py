"""The wave energy converter (WEC) of Exercise 10 as a leap-c-lab style system: model and
differentiable dynamics, Gymnasium environment, linear reactive controllers, the acados OCP and
the differentiable MPC planner."""

from .acados_ocp import J_SCALE, U_SCALE, export_parametric_ocp, phiw_powers, wave_forecast
from .controller import (
    LinearController,
    LinearCtx,
    LQDesign,
    design_lq,
    mean_power,
    optimal_damper,
)
from .dynamics import (
    DiscreteModel,
    WECParams,
    WecDynamics,
    discretise,
    excitation_matrix,
    g_polar,
    polar,
)
from .env import WecEnv, WecEnvConfig
from .interfaces import (
    ControllerFromPlanner,
    CtxInterface,
    CtxType,
    ParameterizedController,
    ParameterizedPlanner,
)
from .planner import WecPlanner, WecPlannerConfig

__all__ = [
    "ControllerFromPlanner",
    "CtxInterface",
    "CtxType",
    "DiscreteModel",
    "J_SCALE",
    "LQDesign",
    "LinearController",
    "LinearCtx",
    "ParameterizedController",
    "ParameterizedPlanner",
    "U_SCALE",
    "WECParams",
    "WecDynamics",
    "WecEnv",
    "WecEnvConfig",
    "WecPlanner",
    "WecPlannerConfig",
    "design_lq",
    "discretise",
    "excitation_matrix",
    "export_parametric_ocp",
    "g_polar",
    "mean_power",
    "optimal_damper",
    "phiw_powers",
    "polar",
    "wave_forecast",
]
