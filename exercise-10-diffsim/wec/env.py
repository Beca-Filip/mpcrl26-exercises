"""Gymnasium environment of the WEC: a single plant stepped through `WecDynamics`."""

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch

from .dynamics import WECParams, WecDynamics, discretise


@dataclass(kw_only=True)
class WecEnvConfig:
    """Configuration of the WEC environment.

    Attributes:
        params: hydrodynamic parameters of the buoy and the regular wave.
        dt: sample time [s] of the zero-order hold on the PTO force.
        g_ex: the plant's excitation coefficient (g1, g2); (1, 0) is the nominal long-wave model.
        x_max: stroke limit [m]; exceeding it is reported in info["task"], not enforced.
        u_max: PTO force limit [N], the bounds of the action space.
        periods: episode length in wave periods, after which the episode is truncated.
    """

    params: WECParams = field(default_factory=WECParams)
    dt: float = 0.2
    g_ex: tuple[float, float] = (1.0, 0.0)
    x_max: float = 1.5
    u_max: float = 300e3
    periods: int = 20


class WecEnv(gym.Env):
    """Heaving point absorber in a regular wave with full state information.

    Observation: [x, v, w1, w2], buoy heave [m] and velocity [m/s] and the wave state
    w = A (cos, sin)(omega t) (wave elevation eta = w1). Action: the PTO force u [N], shape (1,).
    Reward: the electrical energy delivered over the sample [J], the exact integral of
    -u v - r u^2. Every episode starts from rest at a wave crest, z = 0 and w = A (1, 0), and is
    truncated after `periods` wave periods; it never terminates. `info` carries the excitation
    force `Fex` [N] acting during the sample, the mean electrical power `Pe` [W] of the sample and
    the time `t` [s] at its start; at the end of the episode `info["task"]` reports whether the
    stroke limit was violated. The dynamics are the exact sampled-data model `WecDynamics`, the
    same module a training loop differentiates through.
    """

    metadata = {"render_modes": []}

    def __init__(self, render_mode: str | None = None, cfg: WecEnvConfig | None = None):
        self.cfg = WecEnvConfig() if cfg is None else cfg
        self.render_mode = render_mode
        self.p = self.cfg.params
        self.g_ex = np.asarray(self.cfg.g_ex, dtype=float)
        self.d = discretise(self.p, self.cfg.dt).with_excitation(self.g_ex)
        self.n_per = int(round(self.p.wave_period / self.cfg.dt))   # samples per wave period
        self.max_steps = self.cfg.periods * self.n_per
        self.dynamics = WecDynamics(self.d, self.p.amp)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64)
        self.action_space = gym.spaces.Box(
            -self.cfg.u_max, self.cfg.u_max, shape=(1,), dtype=np.float64
        )
        self.obs: torch.Tensor | None = None   # (1, 4)
        self.t, self.k, self.x_abs_max = 0.0, 0, 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.obs = self.dynamics.reset(torch.zeros(1, dtype=torch.float64))
        self.t, self.k, self.x_abs_max = 0.0, 0, 0.0
        return self.obs[0].numpy().copy(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        u = torch.as_tensor(np.asarray(action, dtype=float).reshape(1, 1))
        w = self.obs[0, 2:].numpy()
        info = {"t": self.t, "Fex": float(self.p.gamma * (self.g_ex @ w))}
        with torch.no_grad():
            self.obs, energy = self.dynamics.step(self.obs, u)
        energy = float(energy[0])
        info["Pe"] = energy / self.d.dt
        self.t += self.d.dt
        self.k += 1
        self.x_abs_max = max(self.x_abs_max, abs(float(self.obs[0, 0])))
        truncated = self.k >= self.max_steps
        if truncated:
            violation = self.x_abs_max > self.cfg.x_max
            info["task"] = {"violation": violation, "success": not violation}
        return self.obs[0].numpy().copy(), energy, False, truncated, info
