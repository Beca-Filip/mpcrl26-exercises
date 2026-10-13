# Exercise 10: differentiable simulation with a leap-c MPC policy

A wave energy converter whose economic MPC has a wrong excitation model. The MPC becomes a
differentiable PyTorch policy with leap-c, the plant a differentiable torch function, and the
MPC's excitation coefficient is trained with SHAC and truncated BPTT. About 90 minutes.

Files: `exercise.py` (marimo notebook), `solution.py`, `utils.py` (all domain code: model,
LQ design, acados OCP, evaluation, plots).

## Requirements beyond the root environment

`uv sync --all-extras` (or `uv sync --extra diffsim`) installs all Python packages, including
`leap-c` (`217e1c8`, which provides `leap_c.torch.AcadosDiffMpcTorch` and
`leap_c.parameters.AcadosParameterManager`), `acados_template` and `casadi` 3.7. Both are pinned
git sources in `pyproject.toml` because neither is on PyPI.

What `uv sync` cannot install is the compiled acados library and the `t_renderer` binary.
On Linux x86-64, build them once into `external/acados` (needs `git`, `cmake`, a C compiler,
`curl` and `uv`). The script also runs `uv sync --all-extras`:

```bash
bash scripts/setup_acados.sh
```

The script pins acados to `6111f01f2220a241057b4d56d8fe3a2cddd0309f`. On other platforms, follow
its steps by hand with the matching `t_renderer` from the
[tera_renderer releases](https://github.com/acados/tera_renderer/releases/tag/v0.2.0).
The script builds acados with an `$ORIGIN` RPATH, so no `LD_LIBRARY_PATH` is needed, and adds
`ACADOS_SOURCE_DIR` to the repository's `.env`. marimo loads that file into every kernel, so
`uv run marimo edit` and the VS Code marimo extension find acados with no further setup. Plain
Python scripts that use acados need the variables in the shell:

```bash
source scripts/env.sh
```

The notebook compiles six small acados solvers into temporary directories (about 2 s each) and
trains three policies (about 30 s each on a laptop CPU; the iteration budget is adjustable).

```bash
uv run marimo edit exercise-10-diffsim/exercise.py
```
