# Contributing to CBTV

Thank you for helping. Bug reports, documentation fixes and code are all welcome. Please follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Development install

```
git clone https://github.com/MysteriousPenguin57/CBTV
cd CBTV
pip install -e ".[gpu]"        # or: pip install -e .   (CPU only)
```

Python 3.10 or newer. The `[gpu]` extra installs CuPy for CUDA 12; with CUDA 11 install `cupy-cuda11x`
yourself instead. CuPy's wheel needs a CUDA 12 runtime (system toolkit or conda-forge packages, with `CUDA_PATH` set).

To reproduce the reference outputs exactly (needed for the golden tests), use a clean environment with
`PYTHONNOUSERSITE=1`, install `requirements-lock.txt`, then `pip install -e . --no-deps` and `pip check`. Full steps,
including the GPU runtime, are in [docs/install.md](docs/install.md).

## Running the tests

```
pip install -e ".[test]"       # pytest, pandas, hypothesis (the test suite's own deps)
pytest -m "not gpu"            # CPU tests: run these before every pull request
pytest -m gpu                  # GPU tests: need a CUDA GPU; run them locally
```

GPU tests are marked `@pytest.mark.gpu` and are skipped without CuPy and a CUDA GPU. Tests that run the tracker
on a fixture are marked `slow`; `pytest -m "not gpu and not slow"` is a quick check without them.

On some Windows machines pytest cannot write to its default temporary folder (`PermissionError` under
`%TEMP%\pytest-of-<user>`). Point it elsewhere:

```
pytest -m "not gpu" --basetemp <some writable folder> -p no:cacheprovider
```

The regression fixtures can also be run directly:

```
python tools/run_fixtures.py cpu <outdir> --bare-env
python tools/run_fixtures.py gpu <outdir>
```

Each writes `<outdir>/<fixture>/tracks.csv`; compare it with `tests/fixtures/<fixture>/golden_gpu_tracks.csv`.

## The golden rule: fixtures stay byte-identical

`tests/fixtures/*/golden_gpu_tracks.csv` are the reference outputs of the published configuration.
A change must leave every fixture output **byte-identical** to its golden file, on CPU and on GPU, unless
the change is meant to alter results.

- New behaviour goes behind a flag that is off by default.
- If a golden file changes, do not edit it to make the test pass. Stop and report the difference.
- Goldens are regenerated only with an agreed issue that explains why results change, and the
  new files are committed together with that explanation.

## Workflow

- One branch per change, named after the issue (for example `issue-12-error-messages`). Keep changes small and
  focused; no drive-by refactors.
- Match the style, naming and comment density of the surrounding code.
- Describe in the pull request what changed, the commands you ran and their results (tests, fixtures).
- Never commit large run outputs (`*.mp4`, run directories); `.gitignore` excludes most of them.

## Reporting a bug

Open an issue with the exact command, the full output with `--verbose 1`, your OS, Python version, and
whether you ran on CPU or GPU. If you can share a short clip and detection CSV that reproduce it, attach them.
