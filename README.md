# MPCRL26 exercises

Exercises for the [Fall School on Model Predictive Control and Reinforcement Learning](https://www.syscop.de/teaching/ws2026/fall-school-model-predictive-control-and-reinforcement-learning) (University of Freiburg, October 7–16, 2026).

Exercises are released here session by session. Each uses an
`exercise-NN-topic/` folder with a marimo `exercise.py` notebook and supporting
files, matching the organizer materials. Solutions are added later as
`solution.py` in the same folder for self-study.

Choose one of the two setup routes below.

## 1. Local Python environment

**Requirements:** [Git](https://git-scm.com/install/), [Python](https://www.python.org/downloads/) (3.11 recommended), and [uv](https://docs.astral.sh/uv/getting-started/installation/) or pip.

1. Clone this repository and open a terminal in its root directory.
2. Install the dependencies and start the marimo editor:

   ```bash
   uv python install 3.11
   uv sync
   uv run marimo edit
   ```

   With pip instead:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
   marimo edit
   ```

3. Open a released exercise in the editor. For VS Code, install its **Python**,
   **marimo** and **ty** extensions, select `.venv` as the Python environment, and use
   **marimo: Open as marimo notebook**.

Dependencies are declared in `pyproject.toml`; `uv.lock` pins the versions used
by `uv sync`. PyTorch is installed as the CPU version, which is all the
exercises need.

With an NVIDIA GPU instead (optional, not supported):

```bash
uv sync
uv pip install torch --torch-backend=auto
source .venv/bin/activate
marimo edit
```

## 2. VS Code Dev Container

**Requirements:** [Git](https://git-scm.com/install/), [Docker Desktop](https://www.docker.com/products/docker-desktop/), [VS Code](https://code.visualstudio.com/download), and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

Install Git on your computer before cloning; VS Code provides Git integration
but does not install Git itself.

1. In VS Code, select **Source Control → Clone Repository** (or **Git: Clone**
   in the Command Palette) and enter `https://github.com/mpcrl-school/mpcrl26-exercises.git`.
2. Open the cloned folder and start Docker Desktop.
3. Select **Dev Containers: Reopen in Container** and wait for the build to finish.
4. Open a released exercise with **marimo: Open as marimo notebook**, using the
   container's Python environment. The first time, the marimo extension can
   take a minute to load.

The Dev Container installs Python and the dependencies using the root
`Dockerfile`, and adds the Python, marimo and ty extensions automatically.
Notebook editing and Git integration stay in VS Code.
