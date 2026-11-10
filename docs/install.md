# Installation

CBTV needs Python 3.10 or newer. It has been tested on Windows 11 (CPU and NVIDIA GPU) and on Linux (Ubuntu, CPU).
A GPU is optional: without one, CBTV runs on the CPU and gives the same tracks, more slowly.

The core dependencies are numpy (2.x), scipy, opencv-python, scikit-image and tqdm. They are installed
automatically by each of the methods below.

## With pip, from GitHub

```
pip install "git+https://github.com/MysteriousPenguin57/CBTV.git"
```

This installs the `cbtv` command and the `cbtv` Python package. Check the install with `cbtv --version`.

The examples and test fixtures are not part of the installed package. To run the quick start in the README, also
clone or download the repository.

## From a clone of the repository

```
git clone https://github.com/MysteriousPenguin57/CBTV.git
cd CBTV
pip install .
```

For development, install in editable mode, so that changes to the source take effect without reinstalling:

```
pip install -e ".[test]"
```

Without installing, you can also run `pip install -r requirements.txt` and then `python run_cbtv.py` from the
repository folder. It takes the same options as the `cbtv` command.

## Optional extras

| extra | adds | needed for |
|---|---|---|
| `[gpu]` | `cupy-cuda12x` | GPU acceleration with CUDA 12 (see "GPU" below) |
| `[cellpose]` | `cellpose` | the `cellpose` detector (`--detector cellpose`) |
| `[test]` | `pytest`, `pandas`, `hypothesis` | running the test suite |
| `[api]` | `pandas` | the Python function `cbtv.track()`, which returns a pandas DataFrame |

Extras can be combined, for example `pip install ".[gpu,test]"` from a clone, or
`pip install "cbtv[gpu] @ git+https://github.com/MysteriousPenguin57/CBTV.git"` from GitHub.

## With conda or mamba

`environment.yml` creates an environment called `cbtv` with the core dependencies (CPU only). From a clone of the
repository:

```
conda env create -f environment.yml
conda activate cbtv
pip install --no-deps .
```

With mamba, replace `conda env create` by `mamba env create`. Add `-e` to the last command for an editable install.
Activate the environment before running `cbtv`: some libraries from conda-forge load their DLLs from the
environment's folders, and on Windows a run with the environment's Python but without activation can crash.
CBTV is not on conda-forge yet; `recipe/meta.yaml` is a draft recipe.

## Reproducing the paper's environment exactly

`requirements-lock.txt` pins the exact package versions that gave the reference outputs in `tests/fixtures/` and the
numbers in the paper. Install it into a new, empty environment:

```
python -m venv .venv
.venv\Scripts\activate                        # Linux/macOS: source .venv/bin/activate
set PYTHONNOUSERSITE=1                        # Linux/macOS: export PYTHONNOUSERSITE=1
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pip check                           # should report no broken requirements
```

`PYTHONNOUSERSITE=1` stops packages in your user site-packages (on Windows `%APPDATA%\Python\Python3xx`) from
replacing the pinned numpy, OpenCV or pandas. With this environment, CPU and GPU runs gave byte-identical tracks on
all 71 benchmark cases of the paper on Windows, and Linux (CPU) matched Windows byte for byte on the 10 cases checked.
Linux with a GPU has not been tested. With other library versions, small numerical differences can change individual
links, so the tracks may not match the reference outputs byte for byte.

## GPU

The GPU path uses CuPy. The `[gpu]` extra installs `cupy-cuda12x`, which needs an NVIDIA GPU, a driver for CUDA 12, and
the CUDA 12 runtime libraries. The CuPy wheel does not include the runtime, so provide it in one of two ways.

1. **A system CUDA 12.x toolkit**, with `CUDA_PATH` pointing at it (for example
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`). A `CUDA_PATH` that points at CUDA 13 does not work
   with `cupy-cuda12x`.
2. **conda-forge packages inside a conda environment.** The test environment was set up this way, with CUDA 12.9
   components:

   ```
   conda install -c conda-forge --no-deps "cuda-nvrtc<13" "cuda-cudart<13" "libcublas<13" "libcufft<13" ^
       "libcurand<13" "libcusolver<13" "libcusparse<13" "libnvjitlink<13"
   conda install -c conda-forge --no-deps "cuda-cudart-dev<13" "cuda-nvrtc-dev<13" "cuda-cccl<13" "cuda-crt-dev_win-64<13"
   conda install -c conda-forge --no-deps "cuda-cudart-dev_win-64<13" "cuda-cudart-static_win-64<13"
   python -m pip install cupy-cuda12x==13.6.0
   set CUDA_PATH=%CONDA_PREFIX%\Library                 # Linux: export CUDA_PATH=$CONDA_PREFIX
   ```

   The `*_win-64` packages are for Windows; on Linux, leave them out. `environment-ci.yml` pins the versions that were
   used (cuda-cudart 12.9.79, cuda-nvrtc 12.9.86, cuda-cccl 12.9.27, libcublas 12.9.2.10, libcufft 11.4.1.4,
   libcurand 10.3.10.19, libcusolver 11.7.5.82, libcusparse 12.5.10.65 and libnvjitlink 12.9.86).

With CUDA 11, install `cupy-cuda11x` instead of the `[gpu]` extra.

Check the GPU before tracking. This command should print `0.0` without an error:

```
python -c "import cupy; print(cupy.zeros(1).sum())"
```

If it fails, `cbtv --gpu auto` (the default) runs on the CPU and prints a `[WARN]` line, and `cbtv --gpu 1` stops with
an error.

## Docker

A CPU image can be built from the repository: `docker build -t cbtv .`. See `docker/README.md` for how to run it,
and for notes on a CUDA image.
