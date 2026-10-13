import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import casadi as ca
    import gymnasium as gym
    import numpy as np
    import torch
    from acados_template import AcadosOcp
    from leap_c.parameters import AcadosParameterManager
    from leap_c.torch import AcadosDiffMpcTorch
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecMonitor
    from utils import (
        CartPoleEnv,
        EpisodeReturns,
        cartpole_ode,
        plot_closed_loop,
        plot_reference_effect,
        plot_training,
        rk4,
        rollout,
        rollout_gif,
    )

    return (
        AcadosDiffMpcTorch,
        AcadosOcp,
        AcadosParameterManager,
        CartPoleEnv,
        EpisodeReturns,
        SAC,
        VecMonitor,
        ca,
        cartpole_ode,
        gym,
        make_vec_env,
        mo,
        np,
        plot_closed_loop,
        plot_reference_effect,
        plot_training,
        rk4,
        rollout,
        rollout_gif,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 09 — Learning the reference of an MPC with RL

    **Goal:** let an RL agent choose the reference angle $\theta_\mathrm{ref}$ of a
    myopic MPC, so that the MPC swings up the cart pole of Exercise 07.

    We write the OCP of Exercise 07 again, but with a horizon of only 0.25 s
    ($N=5$). We call this MPC **myopic**, i.e. short-sighted: it is cheap, but it
    does not look far enough ahead to plan the swing-up. Instead of differentiating through the MPC as in Exercise 08, we
    put the MPC **inside the environment**: the agent picks
    $\theta_\mathrm{ref}$, the MPC computes the force, and any RL library can
    train the agent.

    ## 1. The reference angle as a parameter
    Exercise 07 tracked the upright state $x_\mathrm{ref}=(0,0,0,0)$. Here the
    angle reference $\theta_\mathrm{ref}$ becomes a parameter, as in Exercise 08,
    and the cost of every stage penalizes the deviation from
    $x_\mathrm{ref}=(0,\theta_\mathrm{ref},0,0)$:

    $$
    \ell(x,F)=\tfrac12\,\bigl(x-x_\mathrm{ref},\,F\bigr)^\top W\,\bigl(x-x_\mathrm{ref},\,F\bigr).
    $$

    A target away from upright lets the myopic MPC plan swings that it
    would never plan on its own.

    The cell below is the OCP of Exercise 07 with a short horizon. Complete the
    reference in `make_ocp`: register $\theta_\mathrm{ref}$ (`xref1`) with the
    parameter manager and build the state error $x-x_\mathrm{ref}$.
    """)
    return


@app.cell
def _(
    AcadosDiffMpcTorch,
    AcadosOcp,
    AcadosParameterManager,
    ca,
    cartpole_ode,
    np,
    rk4,
):
    def make_ocp(N_horizon=5, T_horizon=0.25, Fmax=80.0):
        ocp = AcadosOcp()
        ocp.solver_options.N_horizon, ocp.solver_options.tf = N_horizon, T_horizon

        # Model of Exercise 07, one RK4 step per stage (leap-c needs discrete dynamics).
        x, u = ca.SX.sym("x", 4), ca.SX.sym("u", 1)
        ocp.model.name, ocp.model.x, ocp.model.u = "cartpole", x, u
        ocp.model.disc_dyn_expr = rk4(cartpole_ode, x, u, T_horizon / N_horizon)
        ocp.solver_options.integrator_type = "DISCRETE"

        # The angle reference theta_ref (xref1) as a parameter.
        manager = AcadosParameterManager(N_horizon=N_horizon)
        xref1 = manager.register_parameter("xref1", default=np.array([[0.0]]), differentiable=True)  # TODO: register the angle reference (one value, default 0) as a differentiable parameter.
        xref = ca.vertcat(0.0, xref1, 0.0, 0.0)
        x_err = x - xref  # TODO: state error x - (0, xref1, 0, 0).

        print(f"ocp.model.x.shape", ocp.model.x.shape)
        print(f"xref1.shape", xref1.shape)
        print(f"x_err.shape", x_err.shape)

        # Cost of Exercise 07 around the reference: 1/2 y^T W y with y = (x - x_ref, F).
        ocp.cost.cost_type = ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.model.cost_y_expr, ocp.model.cost_y_expr_e = ca.vertcat(x_err, u), x_err
        ocp.cost.W = np.diag([1e3, 1e3, 1e-2, 1e-2, 1e-2])
        ocp.cost.W_e = ocp.cost.W[:4, :4]
        ocp.cost.yref, ocp.cost.yref_e = np.zeros(5), np.zeros(4)

        # Force limits. The initial state is set to the measured state in every solve.
        ocp.constraints.lbu, ocp.constraints.ubu = np.array([-Fmax]), np.array([Fmax])
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.qp_tol, ocp.solver_options.tol = 1e-4, 1e-3
        return ocp, manager

    THETA_MAX = np.pi  # the action a in [-1, 1] gives theta_ref = THETA_MAX a^3
    mpc = AcadosDiffMpcTorch(*make_ocp(), num_threads_batch_solver=2)
    return THETA_MAX, make_ocp, mpc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The agent's action $a\in[-1,1]$ sets $\theta_\mathrm{ref}=\pi\,a^3$. With the
    cube, the action space is fine near the upright target and coarse far from
    it: $a=0.2$ shifts the target by only 0.025 rad, while $a=\pm1$ moves it all
    the way down. The plot shows the MPC plan for several actions $a$ in
    three scenarios. Black is the plain MPC with $a=0$.
    """)
    return


@app.cell
def _(THETA_MAX, mpc, np, plot_reference_effect):
    # Starting states (x, theta, v, omega), theta = 0 is upright. Try your own.
    scenarios = {
        "hanging": [0.0, np.pi, 0.0, 0.0],
        "sideways": [0.0, np.pi / 2, 0.0, 0.0],
        "near top": [0.0, 0.3, 0.0, 0.0],
    }
    plot_reference_effect(mpc, THETA_MAX, scenarios)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How does the reference angle change the plan in the three scenarios?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The MPC inside the environment
    A [Gymnasium wrapper](https://gymnasium.farama.org/api/wrappers/) turns the
    cart pole into an environment whose action is $a\in[-1,1]$ with
    $\theta_\mathrm{ref}=\pi\,a^3$. In every step, the MPC computes the force from the current
    state, and `ctx` warm-starts the next solve. The MPC gets the state, the
    agent gets $(x,\sin\theta,\cos\theta,v,\omega)$, which has no jump at the
    hanging state $\theta=\pm\pi$. The diagram shows the loop: for the agent,
    the MPC is simply part of the environment. Complete `step`.
    """)
    return


@app.cell
def _(CartPoleEnv, THETA_MAX, gym, mpc, np, torch):
    class MpcEnv(gym.Wrapper):
        def __init__(self, env, mpc):
            super().__init__(env)
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
            self.mpc = mpc
            self.ctx = None

        def reset(self, **kwargs):
            self.ctx = None
            return self.env.reset(**kwargs)

        def step(self, action):
            theta_ref = torch.tensor([THETA_MAX * action**3], dtype=torch.float32)  # TODO: THETA_MAX a^3 as a float32 tensor of shape (1, 1).
            self.theta_ref = theta_ref.numpy()[0]  # kept for plotting
            state = torch.as_tensor(self.unwrapped.s, dtype=torch.float32)[None]  # the full state
            # print(f"theta_ref.shape", theta_ref.shape, f"theta_ref", theta_ref)
            # print(f"state.shape", state.shape, f"state", state)
            with torch.no_grad():
                # TODO: solve the MPC for state with params {"xref1": theta_ref}, warm-started with self.ctx.
                self.ctx, force, plan_x, _, _ = self.mpc(state, params={"xref1": theta_ref}, ctx=self.ctx)
            self.force, self.plan = force.numpy()[0], plan_x.numpy()[0]
            return self.env.step(self.force)

    def make_env(render_mode=None, random_start=False, mpc=mpc):
        return MpcEnv(CartPoleEnv(render_mode=render_mode, random_start=random_start), mpc)

    return (make_env,)


@app.cell
def _(make_env, np):
    def zero_parameter_policy(obs):
        # Always a = 0, i.e. theta_ref = 0: the plain myopic MPC.
        return np.zeros(1, dtype=np.float32)

    _env = make_env()
    _obs, _ = _env.reset()
    for _steps in range(1, 81):
        _obs, _reward, _terminated, _truncated, _ = _env.step(zero_parameter_policy(_obs))
        if _terminated or _truncated:
            break
    print(f"Plain MPC (a = 0): the episode lasted {_steps} of 80 steps.")
    return (zero_parameter_policy,)


@app.cell
def _(make_env, mo, rollout_gif, zero_parameter_policy):
    mo.image(rollout_gif(make_env, zero_parameter_policy), width=480)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Training with SAC
    We train
    [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
    on 8 copies of the wrapped environment.

    - Half of the training episodes start near upright, the other half at a
      random angle, so the agent sees the box early on. The evaluation below
      starts hanging, as in Exercise 07.
    - The plain MPC with $a=0$ is already a reasonable controller, so the agent
      starts exactly there (zero output layer) and explores only a little
      around it.
    - The reward is 1 for every step inside the box of Exercise 07, plus
      $\tfrac12(1+\cos\theta)\,e^{-(x/0.5)^2}$, which guides the swing-up and
      keeps the cart near the middle.

    `VecMonitor` records the episode returns. This takes a few minutes.
    """)
    return


@app.cell
def _(
    EpisodeReturns,
    SAC,
    VecMonitor,
    make_env,
    make_vec_env,
    mo,
    plot_training,
    torch,
):
    torch.manual_seed(0)
    # SB3 runs 8 copies of the wrapped environment, one after the other.
    _env = VecMonitor(make_vec_env(lambda: make_env(random_start=True), n_envs=8, seed=0))
    model = SAC(
        "MlpPolicy",
        _env,
        learning_rate=1e-3,
        learning_starts=256,  # only one batch of random actions before learning
        ent_coef="auto_0.01",  # entropy coefficient starts at 0.01 instead of 1 ...
        target_entropy=-4.0,  # ... and is tuned towards low noise, also while all rewards are 0
        policy_kwargs=dict(net_arch=[64, 64]),  # small networks
        gradient_steps=16,  # 16 updates per step of the 8 environments
        seed=0,
        verbose=0,
    )

    with mo.status.progress_bar(total=20_000, title="Training MPC-SAC", show_eta=True, show_rate=True) as _bar:
        _returns = EpisodeReturns(_bar)
        model.learn(20_000, callback=_returns)
    plot_training(_returns.returns)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Myopic MPC with RL against a longer MPC
    We compare three closed loops from the hanging state: the myopic MPC alone,
    the myopic MPC with the trained agent, and a nominal MPC with a horizon of
    0.6 s ($N=12$). The plot uses the plotting function of Exercise 07.
    """)
    return


@app.cell
def _(
    AcadosDiffMpcTorch,
    make_env,
    make_ocp,
    model,
    plot_closed_loop,
    rollout,
    zero_parameter_policy,
):
    _nominal = AcadosDiffMpcTorch(*make_ocp(N_horizon=12, T_horizon=0.6))

    def nominal_env(**kwargs):
        return make_env(mpc=_nominal, **kwargs)

    def sac_parameter_policy(obs):
        return model.predict(obs, deterministic=True)[0]

    plot_closed_loop({
        "myopic MPC (N = 5)": rollout(make_env, zero_parameter_policy),
        "myopic MPC with SAC": rollout(make_env, sac_parameter_policy),
        "nominal MPC (N = 12)": rollout(nominal_env, zero_parameter_policy),
    })
    return (sac_parameter_policy,)


@app.cell
def _(make_env, mo, rollout_gif, sac_parameter_policy):
    mo.image(rollout_gif(make_env, sac_parameter_policy), width=480)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Discussion
    - **The nominal MPC is hard to beat here:** it uses the exact model of the
      cart pole. With more training time and recent improvements of off-policy
      RL, e.g. [CrossQ](https://arxiv.org/abs/1902.05605) or the setup of
      [A Walk in the Park](https://arxiv.org/abs/2208.07860), the agent can
      close more of the gap.
    - **Model mismatch:** with a wrong model, e.g. on a real system, a long
      horizon does not help the nominal MPC, while the agent learns from the
      real environment and can compensate part of the error.
    - **Computation:** the myopic MPC solves a smaller problem in every step.
      The agent moves the long-term planning into a cheap network.
    - **Constraints:** whatever the agent outputs, the MPC still enforces the
      force limits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Bonus and resources
    Feel free to play around with, e.g., the horizon of the myopic MPC, the
    starting states of the reference plot, other reference components than the
    angle, or the training settings. A bigger step: write a vectorized
    environment (a stable-baselines3 `VecEnvWrapper`) that solves the MPC for
    all environments in one batch, and compare the training speed.

    **Useful resources:**
    - [`AcadosDiffMpcTorch`](https://github.com/leap-c/leap-c/blob/main/leap_c/torch.py#L29) and [`AcadosParameterManager`](https://github.com/leap-c/leap-c/blob/main/leap_c/parameters/manager.py#L15): the MPC and its parameters, as in Exercise 08
    - [Gymnasium wrappers](https://gymnasium.farama.org/api/wrappers/): how `MpcEnv` builds on the cart pole
    - [SAC in stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html): the algorithm and its settings

    **Bonus sketch: one batched MPC solve for all environments.** Instead of
    wrapping each cart pole, wrap the vectorized environment, so that one call
    solves the MPC for all of them:

    ```python
    class BatchedMpcVecEnv(VecEnvWrapper):  # from stable_baselines3.common.vec_env
        def __init__(self, venv, mpc):
            super().__init__(venv, action_space=gym.spaces.Box(-1.0, 1.0, (1,), np.float32))
            self.mpc, self.ctx = mpc, None

        def reset(self):
            self.ctx = None
            return self.venv.reset()

        def step_async(self, actions):  # actions of all environments, shape (n_envs, 1)
            states = np.array(self.venv.get_attr("s"))  # full states, shape (n_envs, 4)
            theta_ref = THETA_MAX * actions**3
            # One batched solve for all environments.
            self.ctx, forces, *_ = self.mpc(states, params={"xref1": theta_ref}, ctx=self.ctx)
            self.venv.step_async(forces)  # the cart poles only simulate

        def step_wait(self):
            return self.venv.step_wait()  # observations, rewards, dones, infos

    env = BatchedMpcVecEnv(make_vec_env(CartPoleEnv, n_envs=8, env_kwargs={"random_start": True}), mpc)
    ```

    With this we can leverage the parallelization capabilities of leap-c.
    """)
    return


if __name__ == "__main__":
    app.run()
