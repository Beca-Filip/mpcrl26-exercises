"""Domain code for Exercise 10: a heaving wave energy converter (WEC), its energy-maximising LQ
design, the economic MPC as an acados OCP, evaluation helpers and plots.

Everything here is given. The notebook only touches the differentiable plant, the leap-c
parameter handling, the loss and the training loop. Vendored from the wec-lq prototype.

Conventions
    y = [x, v, w1, w2, u]: buoy state z = (x, v), wave state w = A (cos, sin)(omega t), PTO force u
    acting on the buoy. Electrical power P_e = -u v - r u^2; the controller minimises l = -P_e.
    Excitation coefficient g = (g1, g2): F_ex = Gamma (g1 w1 + g2 w2); g = gain (cos phi, sin phi)
    scales the force by gain and lags it by phi; (1, 0) is the nominal model.
"""

from dataclasses import dataclass, replace
from typing import Callable

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import torch
from acados_template import AcadosOcp
from scipy.linalg import expm, solve_discrete_are
from scipy.optimize import minimize_scalar

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


# --------------------------------------------------------------------------- LQ design
@dataclass(frozen=True)
class LinearPolicy:
    """u = -Kz z - Kw w. Kw = 0 gives pure state feedback (e.g. a passive damper)."""
    Kz: np.ndarray   # 1x2
    Kw: np.ndarray   # 1x2

    def __call__(self, z: np.ndarray, w: np.ndarray) -> float:
        return float(-(self.Kz @ z + self.Kw @ w)[0])

    @classmethod
    def damper(cls, c: float) -> "LinearPolicy":
        return cls(np.array([[0.0, c]]), np.zeros((1, 2)))


@dataclass(frozen=True)
class LQDesign:
    policy: LinearPolicy
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
    return LQDesign(policy=LinearPolicy(Kz, Kw), P=P, Y=Y, H=float(H[0, 0]),
                    dare_residual=float(np.abs(res).max() / np.abs(P).max()),
                    cl_eigs=np.linalg.eigvals(Acl))


def mean_power(d: DiscreteModel, pol: LinearPolicy, amp: float) -> float:
    """Exact mean electrical power [W] of a linear policy in periodic steady state."""
    Acl = d.A - d.Bu @ pol.Kz
    L = np.kron(d.Phiw.T, np.eye(2)) - np.kron(np.eye(2), Acl)
    T = np.linalg.solve(L, (d.Gw - d.Bu @ pol.Kw).flatten(order="F")).reshape((2, 2), order="F")
    Xi = np.vstack([T, np.eye(2), -(pol.Kz @ T + pol.Kw)])
    return float(-(amp**2 / 2) * np.trace(Xi.T @ d.Wd @ Xi) / d.dt)


def optimal_damper(d: DiscreteModel, p: WECParams) -> float:
    """Best sampled passive damper u = -c v for the plant d, judged with the exact cost."""
    F, Z, r = p.F_amp, p.Zi, p.r
    guess = minimize_scalar(lambda c: -0.5 * c * (1 - r * c) * F**2 / abs(Z + c) ** 2,
                            bounds=(0, 1 / r), method="bounded").x   # continuous-time optimum
    res = minimize_scalar(lambda c: -mean_power(d, LinearPolicy.damper(c), p.amp),
                          bounds=(0.2 * guess, 2.0 * guess), method="bounded",
                          options=dict(xatol=1e-6 * guess))
    return float(res.x)


# --------------------------------------------------------------------------- numpy plant
Policy = Callable[[np.ndarray, np.ndarray], float]


class RegularWaveWEC:
    """Exact sampled-data simulator: buoy + regular wave, ZOH PTO force, exact energy accounting.

    Full information: buoy state z = [x, v] and wave state w = [w1, w2], eta = w1.
    g_ex is the plant's excitation coefficient (see DiscreteModel.with_excitation).
    """

    def __init__(self, p: WECParams, dt: float, g_ex=(1.0, 0.0)):
        self.p = p
        self.g_ex = np.asarray(g_ex, dtype=float)
        self.d: DiscreteModel = discretise(p, dt).with_excitation(self.g_ex)
        self.n_per = int(round(p.wave_period / dt))   # samples per wave period
        self.reset()

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.z = np.zeros(2)
        self.w = self.p.amp * np.array([1.0, 0.0])
        self.t = 0.0
        return self.z.copy(), self.w.copy()

    def step(self, u: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Advance one sample. Returns (z, w, electrical energy delivered over the step [J])."""
        y = np.concatenate([self.z, self.w, [u]])
        energy = -float(y @ self.d.Wd @ y)
        self.z = self.d.A @ self.z + self.d.Gw @ self.w + self.d.Bu[:, 0] * u
        self.w = self.d.Phiw @ self.w
        self.t += self.d.dt
        return self.z.copy(), self.w.copy(), energy


def simulate(env: RegularWaveWEC, policy: Policy, n_steps: int) -> dict[str, np.ndarray]:
    """Closed-loop rollout from rest. Pe[i] is the mean electrical power over sample i [W]."""
    z, w = env.reset()
    log = {k: np.zeros(n_steps) for k in ("t", "x", "v", "u", "Fex", "Pe")}
    for i in range(n_steps):
        u = policy(z, w)
        log["t"][i], log["x"][i], log["v"][i] = env.t, z[0], z[1]
        log["u"][i], log["Fex"][i] = u, env.p.gamma * (env.g_ex @ w)
        z, w, e = env.step(u)
        log["Pe"][i] = e / env.d.dt
    return log


def steady_mean(sig: np.ndarray, n_per: int, n_periods: int = 5) -> float:
    """Mean over the last n_periods wave periods."""
    return float(np.mean(sig[-n_per * n_periods:]))


def fundamental(sig: np.ndarray, n_per: int) -> complex:
    """Complex amplitude of the fundamental of sig over its last period, s(t) = Re{S e^{j omega t}}."""
    e = np.exp(-2j * np.pi * np.arange(n_per) / n_per)
    return complex(2 * (sig[-n_per:] @ e) / n_per)


def phase_deg(ref: np.ndarray, sig: np.ndarray, n_per: int) -> float:
    """Phase of sig relative to ref over the last period [deg]; positive means sig leads."""
    return float(np.degrees(np.angle(fundamental(sig, n_per) / fundamental(ref, n_per))))


def evaluate(policy: Policy, env: RegularWaveWEC, periods: int = 20) -> dict:
    """Closed loop from rest; steady-state numbers are means over the last 5 periods."""
    log = simulate(env, policy, periods * env.n_per)
    return dict(power=steady_mean(log["Pe"], env.n_per), xmax=float(np.abs(log["x"]).max()),
                umax=float(np.abs(log["u"]).max()),
                v_phase=phase_deg(log["Fex"], log["v"], env.n_per), log=log)


# --------------------------------------------------------------------------- excitation coefficient
def g_polar(gain: float, phase_deg: float) -> np.ndarray:
    """g = gain (cos phi, sin phi): excitation force scaled by gain and lagging by phi."""
    return gain * np.array([np.cos(np.radians(phase_deg)), np.sin(np.radians(phase_deg))])


def polar(g) -> tuple[float, float]:
    """(gain, phase lag [deg]) of an excitation coefficient g."""
    g = np.asarray(g, dtype=float)
    return float(np.hypot(g[0], g[1])), float(np.degrees(np.arctan2(g[1], g[0])))


# --------------------------------------------------------------------------- acados OCP
U_SCALE = 1e5   # the OCP decides u / U_SCALE   [1e5 N]
J_SCALE = 1e3   # and minimises energy / J_SCALE [kJ]


def make_ocp(d: DiscreteModel, N: int, g, wave, x_max: float = np.inf, u_max: float = np.inf,
             slack_l1: float = 1e6, slack_l2: float = 1e8, name: str = "wec_mpc") -> AcadosOcp:
    """Economic MPC of the WEC as an acados OCP with the LQ value function as terminal cost.

    g, wave: CasADi symbols from an AcadosParameterManager: the MPC's excitation coefficient
    (must be registered differentiable=True, acados provides parametric sensitivities only for
    p_global) and the per-stage wave state. State z = (x, v), control u / U_SCALE, costs in J_SCALE.
    """
    lq = design_lq(d)
    z, u = ca.SX.sym("z", 2), ca.SX.sym("u", 1)
    wt = ca.vertcat(g[0] * wave[0] + g[1] * wave[1], -g[1] * wave[0] + g[0] * wave[1])  # G(g) w
    y = ca.vertcat(z, wt, u)
    S = np.diag([1.0, 1.0, 1.0, 1.0, U_SCALE])

    ocp = AcadosOcp()
    ocp.model.name = name
    ocp.model.x, ocp.model.u = z, u
    ocp.model.disc_dyn_expr = ca.DM(d.A) @ z + ca.DM(d.Gw) @ wt + ca.DM(d.Bu * U_SCALE) @ u
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
    o.N_horizon, o.tf = N, N * d.dt
    o.cost_scaling = np.ones(N + 1)   # acados would otherwise multiply stage costs by dt
    o.integrator_type = "DISCRETE"
    o.nlp_solver_type = "SQP"         # the problem is a QP: one QP solve per call
    o.hessian_approx = "EXACT"
    o.regularize_method = "NO_REGULARIZE"   # the stage Hessians are indefinite on purpose
    o.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    o.qp_solver_cond_N = N
    o.qp_solver_ric_alg = 0           # classical Riccati: needs only R + B'PB > 0
    o.with_solution_sens_wrt_params = True
    o.with_value_sens_wrt_params = True
    o.tol, o.qp_tol = 1e-8, 1e-9
    o.nlp_solver_max_iter, o.qp_solver_iter_max = 20, 200
    return ocp


def phiw_powers(d: DiscreteModel, N: int) -> np.ndarray:
    """Phiw^k for k = 0..N, shape (N+1, 2, 2)."""
    return np.stack([np.linalg.matrix_power(d.Phiw, k) for k in range(N + 1)])


def wave_forecast(phi_pow: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-stage wave states w_k = Phiw^k w for a batch w of shape (B, 2): returns (B, N+1, 2)."""
    return np.einsum("kij,bj->bki", phi_pow, np.asarray(w, dtype=float))


def power_landscape(mpc, phi_pow: np.ndarray, env: RegularWaveWEC, gains, phases,
                    periods: int = 12) -> dict:
    """Closed-loop mean power of the MPC on the plant env for every (gain, phase) of its
    excitation coefficient. All grid points run as one batch of plants (about 80 000 solves)."""
    d, N = env.d, len(phi_pow) - 1
    G = np.array([g_polar(gn, ph) for gn in gains for ph in phases])          # (B, 2)
    Bsz, n_steps = len(G), periods * env.n_per
    g_t = torch.from_numpy(G)                                                   # differentiable parameter: (B, 2) tensor
    z, w = np.zeros((Bsz, 2)), np.tile(env.reset()[1], (Bsz, 1))
    Pe = np.zeros((Bsz, n_steps))
    for i in range(n_steps):
        with torch.no_grad():
            _, u0, *_ = mpc(torch.from_numpy(z), params={"g": g_t, "wave": wave_forecast(phi_pow, w)})
        u = u0[:, 0].numpy() * U_SCALE
        y = np.concatenate([z, w, u[:, None]], axis=1)
        Pe[:, i] = -np.einsum("bi,ij,bj->b", y, d.Wd, y) / d.dt
        z = z @ d.A.T + w @ d.Gw.T + u[:, None] * d.Bu[:, 0]
        w = w @ d.Phiw.T
    P = Pe[:, -5 * env.n_per:].mean(axis=1).reshape(len(gains), len(phases))
    return dict(gain=np.asarray(gains), phase=np.asarray(phases), P=P)


# --------------------------------------------------------------------------- RL configuration
REWARD_SCALE = 1e5   # [J] per unit of reward


@dataclass
class ShacConfig:
    x_max: float                 # stroke limit [m]: reward penalty and state scaling
    horizon: int = 10            # h: rollout length per iteration [steps]
    n_env: int = 16              # parallel plants = batch size of the MPC solver
    iters: int = 150             # actor updates
    gamma: float = 0.99
    critic: bool = True          # False: truncated BPTT (no terminal value)
    lam: float = 0.95            # TD(lambda)
    actor_lr: float = 0.02       # Adam, decays linearly to 10 %
    critic_lr: float = 1e-3
    critic_epochs: int = 16
    critic_warmup: int = 50      # critic-only iterations before the actor moves
    target_alpha: float = 0.4    # target critic: phi' <- alpha phi' + (1 - alpha) phi
    episode_len: int = 80        # steps between random restarts
    restart_scale: float = 0.7   # restart buoy state: uniform in +-scale (x_max, omega x_max)
    action_noise: float = 0.0    # Gaussian noise on the PTO force while training [N]
    grad_clip: float = 1.0
    viol_weight: float = 1e8     # stroke-violation penalty in the reward [J/m^2]
    seed: int = 0


def summarize(policy, env: RegularWaveWEC) -> dict:
    """Training callback: steady-state power and the policy's excitation coefficient."""
    gain, phase = polar(policy.g.detach().numpy())
    ev = evaluate(policy.act, env)
    return dict(power=ev["power"], xmax=ev["xmax"], gain=gain, phase=phase)


def result_table(rows: list[tuple], P_nom: float, P_orc: float) -> str:
    """Markdown table. rows: (name, power [W], xmax [m], gain, phase [deg]); gain may be None."""
    out = ["| controller | P_e [kW] | of gap | max abs x [m] | gain | phase [deg] |",
           "|---|---|---|---|---|---|"]
    for name, P, xmax, gain, phase in rows:
        gp = f"{gain:.3f} | {phase:.1f}" if gain is not None else "- | -"
        out.append(f"| {name} | {P/1e3:.2f} | {100*(P - P_nom)/(P_orc - P_nom):.0f} % | {xmax:.3f} | {gp} |")
    return "\n".join(out)


# --------------------------------------------------------------------------- plots
C_LQ, C_PAS = "#2a78d6", "#eb6834"
C_MPC, C_NOM, C_SHAC, C_BPTT, C_SHORT = "#1baf7a", "#eda100", "#4a3aa7", "#008300", "#e87ba4"
_KEYS = (("truncated", C_SHORT), ("SHAC", C_SHAC), ("BPTT", C_BPTT), ("nominal", C_NOM),
         ("LQ", C_LQ), ("passive", C_PAS), ("damper", C_PAS))


def _color(label: str) -> str:
    return next((c for key, c in _KEYS if key in label), C_MPC)


def plot_closed_loop(logs: dict[str, dict], x_max: float, u_max: float, n_per: int):
    """Heave over the whole run, PTO force and electrical power over the last two periods."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 9))
    first = next(iter(logs.values()))
    sl = slice(-2 * n_per, None)
    ts = first["t"][sl] - first["t"][sl][0]
    axs[0].axhspan(-x_max, x_max, color="0.9", lw=0, label=f"stroke limit +-{x_max:g} m")
    for label, lg in logs.items():
        axs[0].plot(lg["t"], lg["x"], color=_color(label), lw=1.5, label=label)
    axs[0].set(xlabel="time [s]", ylabel="x [m]", title="Heave")
    for lim in (u_max, -u_max):
        axs[1].axhline(lim / 1e3, color="0.5", lw=0.8)
    for label, lg in logs.items():
        axs[1].step(ts, lg["u"][sl] / 1e3, where="post", color=_color(label), lw=1.5, label=label)
    axs[1].set(xlabel="time in the last two periods [s]", ylabel="u [kN]", title="PTO force")
    axs[2].axhline(0, color="0.5", lw=0.8)
    for label, lg in logs.items():
        axs[2].step(ts, lg["Pe"][sl] / 1e3, where="post", color=_color(label), lw=1.5,
                    label=f"{label}: mean {steady_mean(lg['Pe'], n_per) / 1e3:.1f} kW")
    axs[2].set(xlabel="time in the last two periods [s]", ylabel="P_e [kW]",
               title="Electrical power (negative = motoring)")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    return fig


def plot_phasors(logs: dict[str, dict], n_per: int):
    """Velocity against the plant's excitation force: traces (left) and phasors (right).

    Reactive control wants v in phase with F_ex. The right panel draws F_ex as the reference
    arrow at angle 0 and one velocity arrow per controller at its relative phase.
    """
    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    first = next(iter(logs.values()))
    sl = slice(-2 * n_per, None)
    ts = first["t"][sl] - first["t"][sl][0]
    ax_t.plot(ts, first["Fex"][sl] / np.abs(first["Fex"][-n_per:]).max(), "k--", lw=1.2, label="F_ex (plant)")
    for label, lg in logs.items():
        ax_t.plot(ts, lg["v"][sl] / np.abs(lg["v"][-n_per:]).max(), color=_color(label), lw=1.5, label=f"v, {label}")
    ax_t.set(xlabel="time in the last two periods [s]", ylabel="normalised [-]", ylim=(-1.6, 1.2),
             title="Velocity vs excitation force")
    ax_t.grid(alpha=0.3)
    ax_t.legend(fontsize=8, loc="lower right", ncol=3)

    # Unit-length arrows at the phase of v relative to the plant's F_ex (amplitudes not shown).
    ref = fundamental(first["Fex"], n_per)
    arrows = {label: np.exp(1j * np.angle(fundamental(lg["v"], n_per) / ref)) for label, lg in logs.items()}
    ax_p.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2, color="k"))
    ax_p.text(1.02, 0.0, "F_ex", ha="left", va="center")
    for label, ph in arrows.items():
        ax_p.annotate("", xy=(ph.real, ph.imag), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="->", lw=2, color=_color(label)))
        ax_p.text(1.08 * ph.real, 1.08 * ph.imag, f"{label}: {np.degrees(np.angle(ph)):+.0f} deg",
                  color=_color(label), ha="left" if ph.real >= 0 else "right", va="center", fontsize=8)
    th = np.linspace(0, 2 * np.pi, 200)
    ax_p.plot(np.cos(th), np.sin(th), color="0.8", lw=0.8)
    ax_p.set(xlim=(-1.3, 1.6), ylim=(-1.3, 1.3), aspect="equal", title="Phasors (counter-clockwise = leads)")
    ax_p.axis("off")
    fig.tight_layout()
    return fig


def plot_landscape(land: dict, g_true, paths: dict | None = None, ax=None):
    """Contours of the closed-loop power over (phase, gain) of the MPC's excitation coefficient,
    the nominal start, the true coefficient and optional training paths."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.8))
    if land is not None:
        cs = ax.contour(land["phase"], land["gain"], land["P"] / 1e3,
                        levels=[20, 25, 28, 30, 32, 33, 33.4], colors="0.4", linewidths=0.8)
        ax.clabel(cs, fmt="%g kW", fontsize=7)
        ax.set(xlim=(land["phase"][0], land["phase"][-1]), ylim=(0, land["gain"][-1]))
    for r in (paths or {}).values():
        ev = [e for e in r["log"] if "power" in e]
        ax.plot([e["phase"] for e in ev], [e["gain"] for e in ev], "-o", color=r["color"],
                lw=2, ms=3, label=r["label"])
    ax.plot(0, 1, "o", color=C_NOM, ms=10, mec="w", mew=1.5, label="nominal model (start)")
    gain, phase = polar(g_true)
    ax.plot(phase, gain, "*", color=C_MPC, ms=16, mec="w", mew=1.2, label="true coefficient")
    ax.set(xlabel="phase lag of the excitation force [deg]", ylabel="gain [-]",
           title="Closed-loop power landscape P_e(gain, phase)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    return ax.figure


def plot_training(runs: dict, P_nom: float, P_orc: float, g_true, land: dict | None):
    """Left: steady-state power during training. Right: the paths of g on the power landscape.
    runs: name -> dict(label, color, log) with log entries carrying env_steps, power, gain, phase."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axs[0]
    for P, text in ((P_orc, "MPC with the true excitation model"), (P_nom, "nominal model (start)")):
        ax.axhline(P / 1e3, color="0.5", lw=0.8)
        ax.annotate(text, (1, P / 1e3), xycoords=("axes fraction", "data"), fontsize=8,
                    ha="right", va="bottom", color="0.3")
    for r in runs.values():
        ev = [e for e in r["log"] if "power" in e]
        ax.plot([e["env_steps"] / 1e3 for e in ev], [e["power"] / 1e3 for e in ev], "-o",
                color=r["color"], lw=2, ms=3, label=r["label"])
    ax.set(xlabel="simulated environment steps [thousands]", ylabel="P_e [kW]",
           title="Steady-state power of the MPC during training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    plot_landscape(land, g_true, paths=runs, ax=axs[1])
    fig.tight_layout()
    return fig
