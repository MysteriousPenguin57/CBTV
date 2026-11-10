# CBTV in Docker

## CPU image (built in CI)

```
docker build -t cbtv .
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/data" cbtv my_experiment.mp4 --gpu 0 --output-dir results
```

- `/data` is the working directory inside the container. Mount the folder that holds your video there; relative
  paths (video, `--detections-csv`, `--output-dir`) resolve inside it.
- `--user "$(id -u):$(id -g)"` makes the outputs belong to you on Linux. Without it the container runs as its
  built-in non-root user `cbtv` (uid 1000), which may not be allowed to write to the mounted folder.
- CI builds the image and runs the fixture tutorial in it (primary60, `--gpu 0`). The resulting tracks.csv must be
  byte-identical to `tests/fixtures/primary60/golden_gpu_tracks_bt.csv` (job `docker` in .github/workflows/ci.yml).

## CUDA variant (notes only, not built in CI)

The GPU path needs CuPy built for the host's CUDA major version, and the NVIDIA Container Toolkit on the host.
A minimal variant:

```
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY . /opt/cbtv
RUN pip install --no-cache-dir "/opt/cbtv[gpu]"        # cupy-cuda12x
WORKDIR /data
ENTRYPOINT ["cbtv"]
```

Run with `docker run --gpus all ... cbtv my_experiment.mp4 --gpu 1`. For CUDA 11 use an 11.x base image and
`pip install cupy-cuda11x` instead of the `[gpu]` extra. `--gpu auto` falls back to CPU when no device is visible.
