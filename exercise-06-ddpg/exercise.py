import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import gymnasium as gym
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from collections import deque
    from copy import deepcopy
    from utils import encode_gif, plot_policy_value, plot_training

    return (
        deepcopy,
        deque,
        encode_gif,
        gym,
        mo,
        nn,
        np,
        plot_policy_value,
        plot_training,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 06 — Deep Deterministic Policy Gradient

    **Goal:** extend replay-based value learning to
    continuous actions using a deterministic actor and a critic.
    DQN has a small finite set of actions, so $\max_a Q(s,a)$ is computed by
    evaluating every action. With continuous actions this is no longer
    possible, so DDPG learns an actor $\mu_\theta(s)$ that approximates
    $\arg\max_a Q(s,a)$.

    ## 1. Pendulum and bounded actions
    [`Pendulum-v1`](https://gymnasium.farama.org/environments/classic_control/pendulum/)
    is again a torque-controlled pendulum with $\theta=0$ upright:

    $$
    \dot\theta=\omega,\qquad \dot\omega=15\sin\theta+3u,\qquad u\in[-2,2].
    $$

    Compared with Exercise 01 ($\dot\omega=\sin\theta+u$), gravity and torque
    are scaled and the torque is bounded, so a swing-up needs several swings.
    The environment uses semi-implicit Euler with $\Delta t=0.05$ and clips
    $\omega$ to $[-8,8]$. The agent observes $(\cos\theta,\sin\theta,\omega)$
    and receives the reward $r=-\ell(s,u)$ with

    $$
    \ell(s,u)=\theta^2+0.1\,\omega^2+0.001\,u^2,
    $$

    where $\theta$ is wrapped to $[-\pi,\pi)$. Larger return is thus better.
    A rollout ends after 200 steps by a time limit, not by reaching a terminal
    goal. Inspect `Actor` and `Critic` below. The actor scales `tanh`
    outputs to the action bounds, and the critic outputs one scalar per transition.
    """)
    return


@app.cell
def _(gym):
    _env = gym.make("Pendulum-v1")
    print("Observation space:", _env.observation_space)
    print("Action space:", _env.action_space)
    print("Reset:", _env.reset(seed=0))
    _env.close()
    return


@app.cell
def _(nn, torch):
    class Actor(nn.Module):
        def __init__(self, state_dim, low, high):
            super().__init__()
            action_dim = len(low)
            self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim), nn.Tanh())
            # Bounds travel with the network but are not trainable parameters.
            self.register_buffer("center", torch.as_tensor((high + low) / 2, dtype=torch.float32))
            self.register_buffer("scale", torch.as_tensor((high - low) / 2, dtype=torch.float32))

        def forward(self, states):
            return self.center + self.scale * self.net(states)

    class Critic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

        def forward(self, states, actions):
            # Concatenate features, preserving the batch dimension: (B, state_dim+action_dim).
            return self.net(torch.cat([states, actions], dim=-1))

    return Actor, Critic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Exploration
    Perturb the actor action with Gaussian noise and clip it to the bounds:

    $$
    a=\operatorname{clip}\big(\mu_\theta(s)+\varepsilon,\ \texttt{low},\ \texttt{high}\big),
    \qquad \varepsilon\sim\mathcal N(0,\sigma^2).
    $$

    Here `sigma` is the standard deviation in torque units. Return an array
    of the same shape as `action`.
    """)
    return


@app.function
def explore(action, low, high, sigma, rng):
    perturbation = rng.normal(0.0, sigma, size=action.shape)
    return ...  # TODO: add noise, then clip to [low, high].


@app.cell
def _(np):
    _rng = np.random.default_rng(0)
    _low, _high = np.array([-2.0]), np.array([2.0])
    print("Small noise:", explore(np.array([0.5]), _low, _high, 0.1, _rng))
    print("Large noise, clipped:", [explore(np.array([0.5]), _low, _high, 10.0, _rng) for _ in range(3)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question:** Why does a deterministic actor need exploration noise?
    """).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Critic update
    Using the target actor and critic, compute

    $$
    y=r+\gamma(1-d)Q_{\bar w}(s',\mu_{\bar\theta}(s')).
    $$

    Targets, rewards, terminal flags, and critic predictions all have shape
    `(B,1)`. Compute targets without autograd, then minimize the mean squared
    error against $Q_w(s,a)$. Mask **termination**, not time-limit truncation.
    """)
    return


@app.cell
def _(torch):
    def make_targets(rewards, terminated, next_states, target_actor, target_critic, gamma):
        # Target actions and values are labels, not part of either optimizer's graph.
        with torch.no_grad():
            next_actions = target_actor(next_states)
            next_values = target_critic(next_states, next_actions)
            return ...  # TODO: masked, bootstrapped target with shape (B,1).

    def critic_loss(critic, states, actions, targets):
        prediction = critic(states, actions)
        return ...  # TODO: mean squared error between prediction and targets.

    return critic_loss, make_targets


@app.cell
def _(critic_loss, make_targets, torch):
    # Toy networks: the target actor returns 1, the target critic returns 2a.
    _target_actor = lambda s: torch.ones(len(s), 1)
    _target_critic = lambda s, a: 2 * a
    _targets = make_targets(torch.tensor([[1.0], [1.0]]), torch.tensor([[0.0], [1.0]]),
                            torch.zeros(2, 3), _target_actor, _target_critic, gamma=0.5)
    print("Targets (expected [[2.0], [1.0]]):", _targets)
    _loss = critic_loss(lambda s, a: a, torch.zeros(2, 3), torch.tensor([[1.0], [3.0]]), torch.zeros(2, 1))
    print("Critic loss (expected 5.0):", _loss)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Actor update
    Pendulum uses rewards, so the actor **maximizes** the critic. Minimize

    $$
    L_\mu(\theta)=-\frac1B\sum_i Q_w(s_i,\mu_\theta(s_i)).
    $$

    The training loop freezes critic **parameters** during the actor step but
    keeps the gradient through the action input:

    $$
    \nabla_\theta Q_w(s,\mu_\theta(s))=
    \nabla_a Q_w(s,a)|_{a=\mu_\theta(s)}\,\nabla_\theta\mu_\theta(s).
    $$
    """)
    return


@app.function
def actor_loss(actor, critic, states):
    actions = actor(states)
    values = critic(states, actions)
    return ...  # TODO: loss whose minimization maximizes the mean value.


@app.cell
def _(torch):
    _actor = lambda s: torch.full((len(s), 1), 2.0)
    _critic = lambda s, a: 3 * a
    print("Actor loss (expected -6.0):", actor_loss(_actor, _critic, torch.zeros(4, 3)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question:** Why would wrapping the actor loss in `torch.no_grad()` break actor learning?
    """).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Target networks and training
    Implement soft updates for both target networks:

    $$
    \begin{aligned}
    \bar w&\leftarrow(1-\tau)\bar w+\tau w,\\
    \bar\theta&\leftarrow(1-\tau)\bar\theta+\tau\theta.
    \end{aligned}
    $$

    Use in-place updates without autograd. A small $\tau=0.005$ gives slow tracking.

    Read the supplied `train` loop. It starts with random actions for 1,000
    steps, stores transitions in replay, and then alternates critic and actor
    optimization followed by soft updates. Target networks start as exact copies.
    """)
    return


@app.cell
def _(torch):
    @torch.no_grad()
    def soft_update(target, online, tau):
        for target_parameter, online_parameter in zip(target.parameters(), online.parameters()):
            # Mutate target weights without recording an autograd graph.
            target_parameter.lerp_(online_parameter, ...)  # TODO: interpolation weight.

    return (soft_update,)


@app.cell
def _(nn, soft_update, torch):
    _online, _target = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        _online.weight.fill_(1.0)
        _target.weight.fill_(0.0)
    soft_update(_target, _online, 0.5)
    print("Target weight (expected 0.5):", _target.weight.item())
    return


@app.cell
def _(np, torch):
    def sample_batch(buffer, batch_size, rng):
        indices = rng.choice(len(buffer), size=batch_size, replace=False)
        fields = list(zip(*(buffer[int(i)] for i in indices)))
        states, actions, rewards, next_states, terminated = [
            torch.as_tensor(np.asarray(field), dtype=torch.float32) for field in fields]
        # Keep all scalar batch quantities as columns, matching the critic's (B,1) output.
        return states, actions, rewards[:, None], next_states, terminated[:, None]

    return (sample_batch,)


@app.cell
def _(Actor, Critic, deepcopy, deque, gym, np, sample_batch, torch):
    def train(make_targets, critic_loss, actor_loss, soft_update, explore,
              steps=30000, seed=0, noise=0.2, gamma=0.99, tau=0.005, batch_size=64):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        env = gym.make("Pendulum-v1")
        low, high = env.action_space.low, env.action_space.high
        actor = Actor(env.observation_space.shape[0], low, high)
        critic = Critic(env.observation_space.shape[0], env.action_space.shape[0])
        # Initialize both targets from the online networks, then update them only softly.
        target_actor = deepcopy(actor).requires_grad_(False)
        target_critic = deepcopy(critic).requires_grad_(False)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        replay = deque(maxlen=100000)
        returns, q_losses, actor_losses = [], [], []
        total = 0.0
        try:
            state, _ = env.reset(seed=seed)
            for step in range(steps):
                if step < 1000:
                    action = rng.uniform(low, high).astype(np.float32)
                else:
                    with torch.no_grad():
                        nominal = actor(torch.as_tensor(state, dtype=torch.float32)).numpy()
                    action = np.asarray(explore(nominal, low, high, noise, rng), dtype=np.float32)
                next_state, reward, terminated, truncated, _ = env.step(action)
                # Store the executed noisy action, not the actor's nominal action.
                replay.append((state.copy(), action.copy(), reward, next_state.copy(), float(terminated)))
                if len(replay) >= max(1000, batch_size):
                    bs, ba, br, bns, bt = sample_batch(replay, batch_size, rng)
                    targets = make_targets(br, bt, bns, target_actor, target_critic, gamma)
                    loss_q = critic_loss(critic, bs, ba, targets)
                    critic_optimizer.zero_grad()
                    loss_q.backward()
                    critic_optimizer.step()
                    # Freeze critic weights, but preserve gradients through its action input.
                    critic.requires_grad_(False)
                    loss_actor = actor_loss(actor, critic, bs)
                    actor_optimizer.zero_grad()
                    loss_actor.backward()
                    actor_optimizer.step()
                    critic.requires_grad_(True)
                    soft_update(target_actor, actor, tau)
                    soft_update(target_critic, critic, tau)
                    q_losses.append(float(loss_q.detach()))
                    actor_losses.append(float(loss_actor.detach()))
                total += reward
                state = next_state
                # Truncation ends the episode but does not make the stored transition terminal.
                if terminated or truncated:
                    returns.append(total)
                    total = 0.0
                    state, _ = env.reset()
        finally:
            env.close()
        return actor, critic, np.asarray(returns), np.asarray(q_losses), np.asarray(actor_losses)

    return (train,)


@app.cell
def _(mo):
    steps = mo.ui.number(start=2000, stop=200000, step=2000, value=30000, label="Environment steps")
    seed = mo.ui.number(start=0, stop=100, value=0, label="Seed")
    noise = mo.ui.slider(0.0, 1.0, step=0.05, value=0.2, label="Exploration noise sigma")
    run = mo.ui.run_button(label="Train DDPG")
    mo.vstack([mo.hstack([steps, seed, noise]), run])
    return noise, run, seed, steps


@app.cell
def _(
    critic_loss,
    make_targets,
    mo,
    noise,
    plot_training,
    run,
    seed,
    soft_update,
    steps,
    train,
):
    mo.stop(not run.value)
    actor, critic, returns, q_losses, actor_losses = train(make_targets, critic_loss, actor_loss, soft_update, explore,
        steps=int(steps.value), seed=int(seed.value), noise=noise.value)
    plot_training(returns, q_losses, actor_losses)
    return actor, critic


@app.cell
def _(encode_gif, gym, np, torch):
    def evaluate(actor, episodes=5, seed=10000, render=False):
        env = gym.make("Pendulum-v1", render_mode="rgb_array" if render else None)
        rng = np.random.default_rng(seed)
        returns, frames = [], []
        try:
            for episode in range(episodes):
                state, _ = env.reset(seed=seed + episode)
                total = 0.0
                while True:
                    if render and episode == 0:
                        frames.append(env.render())
                    if actor is None:
                        action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
                    else:
                        # Evaluate the deterministic actor: no exploration noise or updates.
                        with torch.no_grad():
                            action = actor(torch.as_tensor(state, dtype=torch.float32)).numpy()
                    state, reward, terminated, truncated, _ = env.step(action)
                    total += reward
                    if terminated or truncated:
                        break
                returns.append(total)
        finally:
            env.close()
        return np.asarray(returns), encode_gif(frames)

    return (evaluate,)


@app.cell
def _(actor, evaluate, mo):
    _learned, _gif = evaluate(actor, render=True)
    _random, _ = evaluate(None)
    print("Deterministic evaluation returns:", _learned, "mean:", _learned.mean())
    print("Random-policy returns on the same initial-state seeds:", _random)
    mo.image(_gif) if _gif else mo.md("No rollout frames.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The final policy and the critic's value $V(s)\approx Q_w(s,\mu_\theta(s))$
    over angle and angular velocity, with $\theta=0$ upright. Compare them with
    the value iteration results of Exercise 03. Here the value is a discounted
    return, i.e. a negative cost-to-go, so larger is better.
    """)
    return


@app.cell
def _(actor, critic, plot_policy_value):
    plot_policy_value(actor, critic)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question:** How can approximation errors of the critic mislead the actor?
    """).callout(kind="info")
    return


if __name__ == "__main__":
    app.run()
