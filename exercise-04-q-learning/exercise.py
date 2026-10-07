import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import gymnasium as gym
    import numpy as np
    from utils import encode_gif, plot_training, plot_policy

    return encode_gif, gym, mo, np, plot_policy, plot_training


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 04 — Tabular Q-learning

    **Goal:** In this exercise, you implement tabular Q-learning and use it to
    learn a policy for the CliffWalking gridworld.

    In Exercise 03 the model $f$ was known, and value iteration computed $V$
    from it. Q-learning needs no model: it learns $Q(s,a)$ directly from the
    transitions $S_0, A_0, R_0, S_1, A_1, R_1, \dotsc$ that the agent observes
    while interacting with the environment.

    ### Setup
    [CliffWalking](https://gymnasium.farama.org/environments/toy_text/cliff_walking/)
    is a deterministic 4×12 gridworld with the start bottom left and the
    terminal goal bottom right. Each step yields reward $-1$; entering the cliff
    between them yields $-100$ and resets the agent to the start. The optimal
    return is $-13$. Gymnasium uses rewards $R_t=-\ell(S_t,A_t)$, so every
    $\min$ of the lecture becomes a $\max$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The Gymnasium interface
    [Gymnasium](https://gymnasium.farama.org/) is the standard RL environment
    interface, also used in Exercises 05 and 06
    ([API](https://gymnasium.farama.org/api/env/)):

    ```python
    class Env:
        observation_space: Space  # all possible states, here Discrete(48)
        action_space: Space       # all possible actions, here Discrete(4)

        def reset(self, seed=None):
            '''Start a new episode. Returns (state, info).'''

        def step(self, action):
            '''Apply an action. Returns (next_state, reward, terminated, truncated, info).'''
    ```

    `reset` starts an episode and returns the initial state and an `info`
    dictionary. `step` advances one time step and returns the transition.
    `terminated` signals a terminal MDP state (the goal), `truncated` an
    external cutoff such as a time limit. Termination implies that the next state value is $0$. This, will be later used for the Q-learning update.

    **States:** The state is the agent's cell, indexed row-major as $s=12i+j$ for row
    $i\in\{0,\dots,3\}$ and column $j\in\{0,\dots,11\}$: start 36, goal 47,
    cliff 37–46.
    **Actions:** 0 up, 1 right, 2 down, 3 left.
    """)
    return


@app.cell
def _(gym, mo):
    _env = gym.make("CliffWalking-v1", render_mode="rgb_array")
    print("Spaces:", _env.observation_space, _env.action_space)
    print("Reset:", _env.reset(seed=0))
    print("Step upward:", _env.step(0))
    _image = _env.render()
    _env.close()
    mo.image(_image, width=480)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One episode under a uniformly random policy (500-step limit, first 50
    steps animated).
    """)
    return


@app.cell
def _(encode_gif, gym, mo):
    _env = gym.make("CliffWalking-v1", render_mode="rgb_array", max_episode_steps=500)
    _env.action_space.seed(0)
    _state, _ = _env.reset(seed=0)
    _frames = [_env.render()]
    _total, _steps = 0.0, 0
    while True:
        _action = _env.action_space.sample()
        _state, _reward, _terminated, _truncated, _ = _env.step(_action)
        _total += _reward
        _steps += 1
        if _steps <= 50:
            _frames.append(_env.render())
        if _terminated or _truncated:
            break
    _env.close()
    print(f"Random policy: return {_total:.0f} after {_steps} steps, goal reached: {_terminated}")
    mo.image(encode_gif(_frames), width=480)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Epsilon-greedy action selection

    Given an estimate $Q$ (here the table `Q[s, a]`), the greedy policy
    $\pi_{\rm greedy}(s)\in\operatorname*{arg\,max}_a Q(s,a)$ needs no model. The
    $\epsilon$-greedy policy follows $\pi_{\rm greedy}$ with probability
    $1-\epsilon$ and acts uniformly at random otherwise, so every action keeps
    being tried.

    Complete `select_action` with the
    [NumPy generator](https://numpy.org/doc/stable/reference/random/generator.html)
    `rng` (`rng.random()`, `rng.integers(n)`); for the greedy action, see
    [`np.argmax`](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)
    and [`np.argmin`](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html).
    """)
    return


@app.function
def select_action(Q, state, epsilon, rng):
    if rng.random() < epsilon:
        return ...  # TODO: random action with rng.integers.
    return ...


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check: action 1 maximizes `Q` at state 36, so $\epsilon=0$ always returns 1 and
    $\epsilon=1$ is roughly uniform.
    """)
    return


@app.cell
def _(np):
    _Q = np.zeros((48, 4))
    _Q[36] = [-1.0, 0.0, -2.0, -3.0]
    _rng = np.random.default_rng(0)

    print("epsilon = 0, five calls:", [select_action(_Q, 36, 0.0, _rng) for _ in range(5)])
    print("epsilon = 1, counts per action over 1000 calls:", np.bincount([select_action(_Q, 36, 1.0, _rng) for _ in range(1000)], minlength=4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Q-learning update
    After each transition $(S_t,A_t,R_t,S_{t+1}, D_t)$, with $D_t=1$ iff the episode terminated:

    $$
    Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\Big[R_t+\gamma(1-D_t)\max_{a'}Q(S_{t+1},a')-Q(S_t,A_t)\Big].
    $$

    Complete `update` (in place).
    """)
    return


@app.function
def update(Q, state, action, reward, next_state, terminated, alpha, gamma):
    # Only termination stops bootstrapping, not truncation.
    target = ...  # TODO: TD target.
    Q[state, action] += alpha * (target - Q[state, action])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Check:** $S_t=36$, $A_t=0$ (up), $R_t=-1$, $S_{t+1}=24$ with
    $\max_{a'}Q(24,a')=-2$, $\alpha=0.5$, $\gamma=1$ gives $-1.5$.
    """)
    return


@app.cell
def _(np):
    _Q = np.zeros((48, 4))
    _Q[24] = [-4.0, -2.0, -6.0, -8.0]
    update(_Q, state=36, action=0, reward=-1.0, next_state=24, terminated=False, alpha=0.5, gamma=1.0)
    print("Q[36, up] =", _Q[36, 0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question:** Why is Q-learning an **off-policy** method? Explain with the TD target.
    """).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Training
    The training loop combines both and logs return and length per episode.
    """)
    return


@app.cell
def _(gym, np):
    def train(select_action, update, episodes=500, epsilon=0.1, alpha=0.5, gamma=1.0, seed=0):
        env = gym.make("CliffWalking-v1", max_episode_steps=500)
        rng = np.random.default_rng(seed)
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        rewards, lengths = [], []
        for episode in range(episodes):
            state, _ = env.reset(seed=seed if episode == 0 else None)
            total = 0.0
            for step in range(500):
                action = select_action(Q, state, epsilon, rng)
                next_state, reward, terminated, truncated, _ = env.step(action)
                update(Q, state, action, reward, next_state, terminated, alpha, gamma)
                total += reward
                state = next_state
                if terminated or truncated:
                    break
            rewards.append(total)
            lengths.append(step + 1)
        env.close()
        return Q, np.asarray(rewards), np.asarray(lengths)

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Vary $\epsilon$ with the slider.
    """)
    return


@app.cell
def _(mo):
    epsilon = mo.ui.slider(0.0, 0.5, step=0.05, value=0.1, label="Exploration epsilon")
    epsilon
    return (epsilon,)


@app.cell
def _(epsilon, plot_training, train):
    Q, rewards, lengths = train(select_action, update, epsilon=epsilon.value)
    print("Mean training return of the last 20 episodes:", rewards[-20:].mean())
    print("Estimated start value max_a Q(36, a):", Q[36].max())
    plot_training(rewards, lengths)
    return (Q,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Evaluate the greedy policy
    As a final step, we evaluate the policy that is greedy with respect to the
    learned $Q$, now without exploration and without further updates: we run
    one episode from the start and report its return.
    """)
    return


@app.cell
def _(encode_gif, gym):
    def rollout(Q, seed=1, render=False):
        env = gym.make("CliffWalking-v1", render_mode="rgb_array" if render else None,
                       max_episode_steps=500)
        state, _ = env.reset(seed=seed)
        frames, total = [], 0.0
        for _ in range(500):
            if render:
                frames.append(env.render())
            state, reward, terminated, truncated, _ = env.step(int(Q[state].argmax()))
            total += reward
            if terminated or truncated:
                break
        if render:
            frames.append(env.render())
        env.close()
        return total, terminated, encode_gif(frames) if render else None

    return (rollout,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The plot colours every cell by $\max_a Q(s,a)$ and marks the greedy action
    with an arrow; the animation shows the greedy episode.
    """)
    return


@app.cell
def _(Q, plot_policy, rollout):
    _total, _success, _ = rollout(Q)
    print("Greedy evaluation return:", _total, " goal reached:", _success)
    plot_policy(Q)
    return


@app.cell
def _(Q, mo, rollout):
    _, _, _gif = rollout(Q, render=True)
    mo.image(_gif, width=480)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question:** Why is the training return far below the greedy return?
    Compare $\max_a Q(36,a)$ with the greedy return.
    """).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Extensions:** vary $\epsilon$ and $\alpha$, or use $\gamma<1$. You can introduce new sliders if you want to 🧠
    """)
    return


if __name__ == "__main__":
    app.run()
