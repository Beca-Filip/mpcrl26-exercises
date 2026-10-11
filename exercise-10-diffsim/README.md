# Exercise 10: differentiable simulation with a leap-c MPC policy

A wave energy converter whose economic MPC has a wrong excitation model. The MPC becomes a
differentiable PyTorch policy with leap-c, the plant a differentiable torch function, and the
MPC's excitation coefficient is trained with SHAC and truncated BPTT. About 90 minutes.

Files: `exercise.py` (marimo notebook), `solution.py`, `utils.py` (all domain code: model,
LQ design, acados OCP, evaluation, plots).

## Requirements beyond the root environment

`uv sync` installs all Python packages, including `leap-c` (`217e1c8`, which provides
`leap_c.torch.AcadosDiffMpcTorch` and `leap_c.parameters.AcadosParameterManager`),
`acados_template` and `casadi` 3.7. Both are pinned git sources in `pyproject.toml` because
neither is on PyPI.

What `uv sync` cannot install is the compiled acados library and the `t_renderer` binary.
Build them once (needs `git`, `cmake`, a C compiler and `curl`), outside this repository:

```bash
git clone https://github.com/acados/acados.git ~/acados
cd ~/acados
git checkout 6111f01f2220a241057b4d56d8fe3a2cddd0309f
git submodule update --init --recursive
cmake -S . -B build -DACADOS_WITH_OPENMP=ON -DACADOS_NUM_THREADS=1 \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --target install -j"$(nproc)"
mkdir -p bin
curl -fsSL -o bin/t_renderer \
  https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-linux-amd64
chmod +x bin/t_renderer
```

The `t_renderer` URL is for Linux x86-64; other platforms use the matching file from the
[tera_renderer releases](https://github.com/acados/tera_renderer/releases/tag/v0.2.0).
Then export these in the shell that starts marimo (on macOS use `DYLD_LIBRARY_PATH`):

```bash
export ACADOS_SOURCE_DIR=~/acados
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The notebook compiles six small acados solvers into temporary directories (about 2 s each) and
trains three policies (about 30 s each on a laptop CPU; the iteration budget is adjustable).

```bash
uv run marimo edit exercise-10-diffsim/exercise.py
```
