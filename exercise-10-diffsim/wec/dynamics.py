"""Model of the heaving wave energy converter (WEC): hydrodynamic parameters, the exact
sampled-data discretisation and the plant as a batched, differentiable torch module.

Conventions
    y = [x, v, w1, w2, u]: buoy state z = (x, v), wave state w = A (cos, sin)(omega t), PTO force u
    acting on the buoy. Electrical power P_e = -u v - r u^2; the controller minimises l = -P_e.
    Excitation coefficient g = (g1, g2): F_ex = Gamma (g1 w1 + g2 w2); g = gain (cos phi, sin phi)
    scales the force by gain and lags it by phi; (1, 0) is the nominal model.
    Observation obs = [x, v, w1, w2] (full information), action u [N].
"""

from dataclasses import dataclass, replace

import numpy as np
import torch
from scipy.linalg import expm
from torch import nn

# --------------------------------------------------------------------------- parameters
RHO, G_ACC = 1025.0, 9.81   # sea water density [kg/m^3], gravity [m/s^2]


@dataclass(frozen=True)
class WECParams:
    """Hemispherical buoy in a regular wave. Illustrative long-wave hydrodynamic coefficients."""
    radius: float = 2.5              # hemisphere radius = draft [m]
    wave_period: float = 8.0         # regular wave period [s]
    wave_height: float = 1.0         # crest-to-trough height [m]
    added_mass_ratio: float = 0.75   # m_a / m at the wave frequency
    eps_loss: float = 0.05           # copper-loss factor eps = r |Z_i(omega)|

    @property
    def omega(self) -> float:
        return 2 * np.pi / self.wave_period

    @property
    def amp(self) -> float:
        return self.wave_height / 2

    @property
    def m(self) -> float:
        return RHO * 2 / 3 * np.pi * self.radius**3

    @property
    def m_a(self) -> float:
        return self.added_mass_ratio * self.m

    @property
    def M(self) -> float:
        """Total inertia m + m_a."""
        return self.m + self.m_a

    @property
    def k(self) -> float:
        """Hydrostatic stiffness rho g A_w."""
        return RHO * G_ACC * np.pi * self.radius**2

    @property
    def gamma(self) -> float:
        """Excitation coefficient F_ex = gamma eta (Froude-Krylov, long waves)."""
        return self.k

    @property
    def B(self) -> float:
        """Radiation damping from the Haskind relation."""
        return self.gamma**2 * self.omega**3 / (2 * RHO * G_ACC**3)

    @property
    def natural_period(self) -> float:
        return 2 * np.pi / np.sqrt(self.k / self.M)

    @property
    def Zi(self) -> complex:
        """Intrinsic impedance B + j(omega M - k/omega) at the wave frequency."""
        return self.B + 1j * (self.omega * self.M - self.k / self.omega)

    @property
    def r(self) -> float:
        """Copper-loss coefficient: P_loss = r u^2."""
        return self.eps_loss / abs(self.Zi)

    @property
    def F_amp(self) -> float:
        """Excitation force amplitude [N]."""
        return self.gamma * self.amp


# --------------------------------------------------------------------------- model
IX, IV, IW1, IW2, IU = range(5)   # index map into y


def continuous_model(p: WECParams) -> tuple[np.ndarray, np.ndarray]:
    """Generator F (5x5) of y' = F y and cost weight W (5x5) of l = u v + r u^2 = y' W y."""
    F = np.zeros((5, 5))
    F[IX, IV] = 1.0
    F[IV, IX] = -p.k / p.M
    F[IV, IV] = -p.B / p.M
    F[IV, IW1] = p.gamma / p.M
    F[IV, IU] = 1.0 / p.M
    F[IW1, IW2] = -p.omega
    F[IW2, IW1] = p.omega
    W = np.zeros((5, 5))
    W[IV, IU] = W[IU, IV] = 0.5
    W[IU, IU] = p.r
    return F, W


def excitation_matrix(g) -> np.ndarray:
    """G(g) = [[g1, g2], [-g2, g1]]: the complex excitation coefficient acting on the wave state."""
    return np.array([[g[0], g[1]], [-g[1], g[0]]], dtype=float)


@dataclass(frozen=True)
class DiscreteModel:
    """z+ = A z + Gw w + Bu u,  w+ = Phiw w,  sum_k y_k' Wd y_k = integral of l (exact ZOH)."""
    A: np.ndarray      # 2x2
    Bu: np.ndarray     # 2x1
    Gw: np.ndarray     # 2x2
    Phiw: np.ndarray   # 2x2 rotation by omega dt
    Wd: np.ndarray     # 5x5 exact integrated cost over one sample
    dt: float

    @property
    def Qzz(self): return self.Wd[:2, :2]
    @property
    def Qzw(self): return self.Wd[:2, 2:4]
    @property
    def Nz(self): return self.Wd[:2, 4:5]
    @property
    def Nw(self): return self.Wd[2:4, 4:5]
    @property
    def R(self): return self.Wd[4:5, 4:5]

    def with_excitation(self, g) -> "DiscreteModel":
        """Model with F_ex = Gamma (g1 w1 + g2 w2). Exact because G(g) commutes with Phiw."""
        T = np.eye(5)
        T[2:4, 2:4] = excitation_matrix(g)
        return replace(self, Gw=self.Gw @ T[2:4, 2:4], Wd=T.T @ self.Wd @ T)


def discretise(p: WECParams, dt: float) -> DiscreteModel:
    """Exact ZOH discretisation of dynamics and integrated stage cost (Van Loan, 1978)."""
    F, W = continuous_model(p)
    n = F.shape[0]
    E = expm(np.block([[-F.T, W], [np.zeros((n, n)), F]]) * dt)
    Phi = E[n:, n:]
    Wd = Phi.T @ E[:n, n:]
    Wd = 0.5 * (Wd + Wd.T)
    return DiscreteModel(A=Phi[:2, :2], Bu=Phi[:2, 4:5], Gw=Phi[:2, 2:4],
                         Phiw=Phi[2:4, 2:4], Wd=Wd, dt=dt)


# --------------------------------------------------------------------------- excitation coefficient
def g_polar(gain: float, phase_deg: float) -> np.ndarray:
    """g = gain (cos phi, sin phi): excitation force scaled by gain and lagging by phi."""
    return gain * np.array([np.cos(np.radians(phase_deg)), np.sin(np.radians(phase_deg))])


def polar(g) -> tuple[float, float]:
    """(gain, phase lag [deg]) of an excitation coefficient g."""
    g = np.asarray(g, dtype=float)
    return float(np.hypot(g[0], g[1])), float(np.degrees(np.arctan2(g[1], g[0])))


# --------------------------------------------------------------------------- differentiable plant
class WecDynamics(nn.Module):
    """The sampled-data plant as pure batched tensor functions: the differentiable simulator.

    z' = A z + Gw w + Bu u, w' = Phiw w, with the exact electrical energy -y' Wd y of one sample.
    Stateless: the caller passes obs = (z, w) in and receives obs' and the step energy, and the
    leading dimension is a free batch of plants. `WecEnv` steps a single plant through this module
    under `torch.no_grad()`; a training loop calls it on a batch with autograd recording the
    whole rollout. All tensors are float64.
    """

    def __init__(self, d: DiscreteModel, amp: float):
        super().__init__()
        for name in ("A", "Gw", "Phiw", "Wd"):
            self.register_buffer(name, torch.from_numpy(getattr(d, name)))
        self.register_buffer("Bu", torch.from_numpy(d.Bu[:, 0]))
        self.amp, self.dt = amp, d.dt
        self.n_per = int(round(2 * np.pi / np.arctan2(d.Phiw[1, 0], d.Phiw[0, 0])))   # samples per wave period

    def reset(self, phase: torch.Tensor) -> torch.Tensor:
        """Buoys at rest, waves eta = A cos(omega t + phase) for a batch of phases (B,). Returns obs (B, 4)."""
        w = self.amp * torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)
        return torch.cat([torch.zeros_like(w), w], dim=-1)

    def step(self, obs: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: (B, 4), u: (B, 1) [N]. Returns obs' (B, 4) and the electrical energy of the step (B,) [J]."""
        z, w = obs[:, :2], obs[:, 2:]
        y = torch.cat([obs, u], dim=-1)
        energy = -torch.einsum("bi,ij,bj->b", y, self.Wd, y)   # Exact step energy, one scalar per plant.
        z_next = z @ self.A.T + w @ self.Gw.T + u * self.Bu     # Batched linear dynamics.
        w_next = w @ self.Phiw.T
        return torch.cat([z_next, w_next], dim=-1), energy
