"""Acados-based planner for the WEC: the economic MPC as a differentiable `ParameterizedPlanner`."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from leap_c.diff_mpc.function import AcadosDiffMpcCtx
from leap_c.parameters.utils import broadcast_default_param
from leap_c.torch import AcadosDiffMpcTorch
from leap_c.utils.collate import collate_acados_diff_mpc_ctx

from .acados_ocp import J_SCALE, U_SCALE, export_parametric_ocp, phiw_powers, wave_forecast
from .dynamics import WECParams, discretise
from .interfaces import ParameterizedPlanner


@dataclass(kw_only=True)
class WecPlannerConfig:
    """Configuration of the WEC planner.

    Attributes:
        N_horizon: number of MPC steps; the OCP has N_horizon + 1 nodes.
        dt: sample time [s]; the horizon covers N_horizon * dt seconds.
        x_max: stroke limit [m], a soft constraint with L1 and L2 slack penalties; inf disables it.
        u_max: PTO force limit [N], a hard bound on the input; inf disables it.
        g_default: default excitation coefficient (g1, g2) of the MPC's model.
        params: hydrodynamic parameters of the MPC's model.
        slack_l1: L1 weight of the stroke slack [J/m].
        slack_l2: L2 weight of the stroke slack [J/m^2].
        name: acados model name; give every compiled planner a distinct one.
        n_batch_init: initially supported batch size of the batch OCP solver (leap-c default 256).
        num_threads_batch_solver: threads of the batch OCP solver (leap-c default 4).
        dtype: dtype of the planner's output tensors.
    """

    N_horizon: int = 20
    dt: float = 0.2
    x_max: float = 1.5
    u_max: float = 300e3
    g_default: tuple[float, float] = (1.0, 0.0)
    params: WECParams = field(default_factory=WECParams)
    slack_l1: float = 1e6
    slack_l2: float = 1e8
    name: str = "wec"
    n_batch_init: int | None = None
    num_threads_batch_solver: int | None = None
    dtype: torch.dtype | None = torch.float64


class WecPlanner(ParameterizedPlanner[AcadosDiffMpcCtx]):
    """Acados-based planner for the WEC: the economic MPC of `wec.acados_ocp` in leap-c.

    The observation is the full state [x, v, w1, w2] of `WecEnv`. The buoy state z = (x, v) is
    the initial state of the OCP, passed with its autograd graph (leap-c provides du0/dz0), and
    the wave state w gives the forecast w_k = Phiw^k w over the horizon, an exogenous stage-wise
    parameter that is detached. The differentiable parameter is the excitation coefficient g of
    the MPC's model, `params={"g": tensor (B, 2)}`, default (1, 0); leap-c provides du0/dg from
    the implicit function theorem on the KKT conditions of the converged QP. Actions are returned
    in Newtons and values in Joules.
    """

    cfg: WecPlannerConfig
    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(self, cfg: WecPlannerConfig | None = None, export_directory: Path | None = None):
        super().__init__()
        self.cfg = WecPlannerConfig() if cfg is None else cfg
        self.d = discretise(self.cfg.params, self.cfg.dt)
        ocp, param_manager, param_space, default_param = export_parametric_ocp(
            self.d, self.cfg.N_horizon, x_max=self.cfg.x_max, u_max=self.cfg.u_max,
            g_default=self.cfg.g_default, slack_l1=self.cfg.slack_l1, slack_l2=self.cfg.slack_l2,
            name=self.cfg.name,
        )
        self.diff_mpc = AcadosDiffMpcTorch(
            ocp, param_manager, export_directory=export_directory,
            n_batch_init=self.cfg.n_batch_init,
            num_threads_batch_solver=self.cfg.num_threads_batch_solver,
            dtype=self.cfg.dtype, verbose=False,
        )
        self.param_manager = param_manager
        self._param_space = param_space
        self._default_param = default_param
        self._phi_pow = phiw_powers(self.d, self.cfg.N_horizon)   # Phiw^k, k = 0..N

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        params: dict[str, torch.Tensor | np.ndarray] | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """obs: (B, 4) = [x, v, w1, w2]; action: (B, 1) [N] fixes u0; params: {"g": (B, 2)}.

        Returns (ctx, u0 (B, 1) [N], x (B, N+1, 2), u (B, N, 1) [N], value (B, 1) [J]).
        """
        x0 = obs[:, :2]
        wave = wave_forecast(self._phi_pow, obs[:, 2:].detach().cpu().numpy())   # exogenous: no gradient through the sea
        p = {"wave": wave, **(params or {})}
        u0 = None if action is None else action / U_SCALE
        ctx, u0, x, u, value = self.diff_mpc(x0=x0, u0=u0, params=p, ctx=ctx)
        return ctx, u0 * U_SCALE, x, u * U_SCALE, value * J_SCALE

    def default_param(self, obs: np.ndarray | torch.Tensor | None = None) -> dict[str, np.ndarray]:
        return broadcast_default_param(self._default_param, obs)
