import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import gymnasium as gym
    from collections import deque
    from copy import deepcopy
    import torch
    from torch import nn
    import numpy as np
    from utils import encode_gif, plot_training

    return deepcopy, deque, encode_gif, gym, mo, nn, np, plot_training, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 05 — Deep Q-learning with PyTorch (Solution)

    **Goal:** train a DQN controller for CartPole with continuous observations
    and two discrete actions. Unlike tabular Q-learning, a neural network shares
    parameters across states. Replay and a target network stabilize training.

    ## 1. Q-network
    Implement a multilayer perceptron with two hidden layers of width 64 and
    ReLU activations (see the PyTorch
    [overview of activation functions](https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)).
    Return one Q-value per action, with **no output activation**.
    Support both a single observation `(4,)` and minibatches `(B,4)`.
    """)
    return


@app.cell
def _(nn):
    class QNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, action_dim),
            )

        def forward(self, states):
            # Linear layers preserve any leading batch dimensions.
            return self.net(states)

    return (QNetwork,)


@app.cell
def _(QNetwork, torch):
    _q = QNetwork(4, 2)
    print("Single observation:", _q(torch.zeros(4)).shape)
    print("Minibatch:", _q(torch.zeros(8, 4)).shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question:** Why is there no activation function after the last layer?
    """).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answer:** Q-values are regression outputs and can take any real value.
    A ReLU would cut off negative values, a sigmoid or tanh would bound them,
    while CartPole Q-values grow to about 87 for $\gamma=0.99$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Experience replay
    Store `(state, action, reward, next_state, terminated)` in a bounded deque.
    Copy state arrays so later mutation cannot alter stored transitions.
    The environment's time limit ends an episode but is not a terminal MDP state.
    Inspect `sample_batch` below: state tensors are `(B,4)`, while actions,
    rewards and terminal flags are `(B,)`. Why sample random minibatches?

    **Answer:** consecutive transitions are strongly correlated. Random
    minibatches are closer to the i.i.d. samples SGD assumes, and each
    transition is reused in many updates.
    """)
    return


@app.function
def store(replay, state, action, reward, next_state, terminated):
    # Copy observations: replay should not reference mutable environment arrays.
    transition = (state.copy(), int(action), float(reward), next_state.copy(), bool(terminated))
    replay.append(transition)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sampling uniformly from replay breaks up consecutive trajectories. Notice
    that actions are integer indices while rewards and states are floating point.
    What would happen if an action tensor had the wrong dtype for `gather`?

    **Answer:** `gather` needs integer indices. A float action tensor raises
    a `RuntimeError`.
    """)
    return


@app.cell
def _(np, torch):
    def sample_batch(buffer, batch_size, rng):
        indices = rng.choice(len(buffer), size=batch_size, replace=False)
        # Transpose a list of transitions into one batch per field.
        states, actions, rewards, next_states, terminated = zip(*(buffer[int(i)] for i in indices))
        return (torch.as_tensor(np.asarray(states), dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.long),
                torch.as_tensor(rewards, dtype=torch.float32),
                torch.as_tensor(np.asarray(next_states), dtype=torch.float32),
                torch.as_tensor(terminated, dtype=torch.bool))

    return (sample_batch,)


@app.cell
def _(deque, np, sample_batch):
    _replay = deque(maxlen=3)
    for _k in range(5):
        store(_replay, np.full(4, float(_k)), _k % 2, 1.0, np.full(4, _k + 1.0), _k == 4)
    print("Replay length (maxlen 3):", len(_replay))
    for _x in sample_batch(_replay, 2, np.random.default_rng(0)):
        print(tuple(_x.shape), _x.dtype)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Bootstrapped target and optimization
    Implement

    $$
    y_i=r_i+\gamma(1-d_i)\max_a Q_{\bar w}(s'_i,a).
    $$

    Use `torch.no_grad()`. The targets are a tensor of shape `(B,)`, one value per
    transition. The supplied loop minimizes
    $B^{-1}\sum_i(Q_w(s_i,a_i)-y_i)^2$. Explain why the target is detached and
    why `gather` selects the sampled action value, rather than the max current value.

    **Answer:** the target is a fixed regression label. Gradients through it
    would also move the label, so only $Q_w(s_i,a_i)$ is fitted. The
    transition $(s_i,a_i,r_i,s'_i)$ carries information only about the
    sampled action $a_i$. The max belongs to the next state, in the target.

    Read `train` below: it initializes the target as a copy of the online
    network, waits for 1,000 replay transitions, and copies weights every 100
    environment steps. Exploration decreases from 1 to 0.05.
    """)
    return


@app.cell
def _(torch):
    def make_targets(rewards, next_states, terminated, target_q, gamma):
        # Targets are fixed regression labels during the online-network update.
        with torch.no_grad():
            next_values = target_q(next_states).max(dim=1).values
            return rewards + gamma * (~terminated).float() * next_values

    return (make_targets,)


@app.cell
def _(make_targets, torch):
    # Toy target network with fixed Q-values for two next states.
    _target_q = lambda next_states: torch.tensor([[1.0, 3.0], [2.0, 5.0]])
    _targets = make_targets(torch.tensor([1.0, 1.0]), torch.zeros(2, 4),
                            torch.tensor([False, True]), _target_q, gamma=0.5)
    print("Targets (expected [2.5, 1.0]):", _targets)
    return


@app.cell
def _(deepcopy, deque, gym, nn, np, sample_batch, torch):
    def train(Network, store, make_targets, episodes=200, seed=0, gamma=0.99, batch_size=64):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        env = gym.make("CartPole-v1", max_episode_steps=200)
        q = Network(env.observation_space.shape[0], env.action_space.n)
        # The target begins as an exact copy and is never optimized by autograd.
        target_q = deepcopy(q).requires_grad_(False)
        optimizer = torch.optim.Adam(q.parameters(), lr=1e-3)
        replay = deque(maxlen=50000)
        returns, losses, steps = [], [], 0
        try:
            for episode in range(episodes):
                state, _ = env.reset(seed=seed if episode == 0 else None)
                epsilon = max(0.05, 1.0 - episode / (0.75 * episodes))
                total = 0.0
                while True:
                    if rng.random() < epsilon:
                        action = int(rng.integers(env.action_space.n))
                    else:
                        with torch.no_grad():
                            action = int(q(torch.as_tensor(state, dtype=torch.float32)).argmax().item())
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    store(replay, state, action, reward, next_state, terminated)
                    steps += 1
                    # Collect enough data before the first minibatch update.
                    if len(replay) >= max(1000, batch_size):
                        bs, ba, br, bns, bt = sample_batch(replay, batch_size, rng)
                        targets = make_targets(br, bns, bt, target_q, gamma)
                        # Select Q(s_i, a_i) for the actions stored in the batch.
                        prediction = q(bs).gather(1, ba[:, None]).squeeze(1)
                        loss = nn.functional.mse_loss(prediction, targets)
                        optimizer.zero_grad()  # Clear gradients from the previous update.
                        loss.backward()
                        nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                    if steps % 100 == 0:
                        target_q.load_state_dict(q.state_dict())
                    total += reward
                    state = next_state
                    # Both flags end the rollout, but only termination masks the target.
                    if terminated or truncated:
                        break
                returns.append(total)
        finally:
            env.close()
        return q, np.asarray(returns), np.asarray(losses)

    return (train,)


@app.cell
def _(mo):
    episodes = mo.ui.number(start=20, stop=1000, step=20, value=200, label="Training episodes")
    seed = mo.ui.number(start=0, stop=100, value=0, label="Seed")
    run = mo.ui.run_button(label="Train DQN")
    mo.hstack([episodes, seed, run])
    return episodes, run, seed


@app.cell
def _(QNetwork, episodes, make_targets, mo, plot_training, run, seed, train):
    mo.stop(not run.value)
    q, returns, losses = train(QNetwork, store, make_targets, episodes=int(episodes.value), seed=int(seed.value))
    plot_training(returns, losses)
    return (q,)


@app.cell
def _(encode_gif, gym, np, torch):
    def evaluate(q, episodes=5, seed=10000, render=False):
        env = gym.make("CartPole-v1", max_episode_steps=200,
                       render_mode="rgb_array" if render else None)
        returns, frames = [], []
        try:
            for episode in range(episodes):
                state, _ = env.reset(seed=seed + episode)
                total = 0.0
                while True:
                    if render and episode == 0:
                        frames.append(env.render())
                    # Greedy evaluation uses new seeds, no replay, and no optimizer.
                    with torch.no_grad():
                        action = int(q(torch.as_tensor(state, dtype=torch.float32)).argmax().item())
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
def _(evaluate, mo, q):
    _returns, _gif = evaluate(q, render=True)
    print("Greedy evaluation returns:", _returns, "mean:", _returns.mean())
    mo.image(_gif) if _gif else mo.md("No rollout frames.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Evaluation and extensions
    Evaluate without exploration on seeds different from training. This version
    caps episodes at 200 steps, so return 200 is the maximum. Success within a
    fixed episode count is not guaranteed; compare several seeds and learning curves.
    A small TD loss alone does not establish a good control policy.
    """)
    return


if __name__ == "__main__":
    app.run()
