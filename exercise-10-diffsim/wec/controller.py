"""Linear reactive controllers of the WEC as `ParameterizedController`s: the energy-maximising LQ
law and the passive damper, with their design from the discrete model."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from leap_c.parameters.utils import broadcast_default_param
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize_scalar

from .dynamics import DiscreteModel, WECParams
from .interfaces import ParameterizedController


@dataclass
class LinearCtx:
    """Context of a controller without a solver (`CtxInterface`): every status is 0."""
    status: np.ndarray
    log: dict[str, float] | None = None


class LinearController(ParameterizedController[LinearCtx]):
    """u = -K obs with the gain row K = [Kz, Kw] as parameter: the LQ law u = -Kz z - Kw w, or with
    Kz = (0, c) and Kw = 0 the passive damper u = -c v. Batched: obs (B, 4) -> u (B, 1) [N]."""

    def __init__(self, Kz: np.ndarray, Kw: np.ndarray):
        super().__init__()
        self.Kz, self.Kw = np.asarray(Kz, dtype=float), np.asarray(Kw, dtype=float)   # 1x2 each
        self.register_buffer("K", torch.from_numpy(np.concatenate([self.Kz, self.Kw], axis=1)[0]))

    def forward(self, obs, params=None, ctx=None) -> tuple[LinearCtx, torch.Tensor]:
        """obs: (B, 4); params: gains (B, 4), or None for the designed K. Returns (ctx, u (B, 1))."""
        K = self.K if params is None else params
        u = -(obs * K).sum(dim=-1, keepdim=True)
        return LinearCtx(status=np.zeros(obs.shape[0], dtype=int)), u

    @property
    def param_space(self) -> gym.Space:
        return gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64)

    def default_param(self, obs=None) -> np.ndarray:
        return broadcast_default_param(self.K.numpy(), obs)

    @classmethod
    def damper(cls, c: float) -> "LinearController":
        return cls(np.array([[0.0, c]]), np.zeros((1, 2)))


# --------------------------------------------------------------------------- LQ design
@dataclass(frozen=True)
class LQDesign:
    controller: LinearController
    P: np.ndarray            # 2x2 Riccati solution (indefinite in general)
    Y: np.ndarray            # 2x2 buoy-wave coupling of the value function
    H: float                 # R + B'PB, must be > 0
    dare_residual: float     # relative residual of the DARE
    cl_eigs: np.ndarray      # eigenvalues of A - B Kz


def _solve_stein(Acl: np.ndarray, Phiw: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve Y - Acl' Y Phiw = rhs."""
    n, m = rhs.shape
    L = np.eye(n * m) - np.kron(Phiw.T, Acl.T)
    return np.linalg.solve(L, rhs.flatten(order="F")).reshape((n, m), order="F")


def design_lq(d: DiscreteModel) -> LQDesign:
    """Average-cost LQ with the indefinite energy cost: DARE on z, Stein equation for Y."""
    A, B, Gw, Phiw = d.A, d.Bu, d.Gw, d.Phiw
    P = solve_discrete_are(A, B, d.Qzz, d.R, s=d.Nz)
    H = d.R + B.T @ P @ B
    Kz = np.linalg.solve(H, B.T @ P @ A + d.Nz.T)
    res = d.Qzz + A.T @ P @ A - (A.T @ P @ B + d.Nz) @ Kz - P
    Acl = A - B @ Kz
    Y = _solve_stein(Acl, Phiw, d.Qzw + Acl.T @ P @ Gw - Kz.T @ d.Nw.T)
    Kw = np.linalg.solve(H, B.T @ P @ Gw + B.T @ Y @ Phiw + d.Nw.T)
    return LQDesign(controller=LinearController(Kz, Kw), P=P, Y=Y, H=float(H[0, 0]),
                    dare_residual=float(np.abs(res).max() / np.abs(P).max()),
                    cl_eigs=np.linalg.eigvals(Acl))


def mean_power(d: DiscreteModel, ctrl: LinearController, amp: float) -> float:
    """Exact mean electrical power [W] of a linear controller in periodic steady state."""
    Acl = d.A - d.Bu @ ctrl.Kz
    L = np.kron(d.Phiw.T, np.eye(2)) - np.kron(np.eye(2), Acl)
    T = np.linalg.solve(L, (d.Gw - d.Bu @ ctrl.Kw).flatten(order="F")).reshape((2, 2), order="F")
    Xi = np.vstack([T, np.eye(2), -(ctrl.Kz @ T + ctrl.Kw)])
    return float(-(amp**2 / 2) * np.trace(Xi.T @ d.Wd @ Xi) / d.dt)


def optimal_damper(d: DiscreteModel, p: WECParams) -> float:
    """Best sampled passive damper u = -c v for the plant d, judged with the exact cost."""
    F, Z, r = p.F_amp, p.Zi, p.r
    guess = minimize_scalar(lambda c: -0.5 * c * (1 - r * c) * F**2 / abs(Z + c) ** 2,
                            bounds=(0, 1 / r), method="bounded").x   # continuous-time optimum
    res = minimize_scalar(lambda c: -mean_power(d, LinearController.damper(c), p.amp),
                          bounds=(0.2 * guess, 2.0 * guess), method="bounded",
                          options=dict(xatol=1e-6 * guess))
    return float(res.x)
