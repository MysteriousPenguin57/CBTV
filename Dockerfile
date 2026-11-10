# CBTV, CPU image.  Build:  docker build -t cbtv .
# Run on a video in the current folder:
#   docker run --rm -v "$PWD:/data" cbtv my_experiment.mp4 --output-dir results
# CUDA variant: see docker/README.md (not built in CI).
FROM python:3.12-slim

# opencv-python (the non-headless wheel CBTV depends on) needs libGL + glib at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cbtv
COPY pyproject.toml README.md LICENSE ./
COPY cbtv ./cbtv
COPY run_cbtv.py cbtv_clean.py render_states.py ./
RUN pip install --no-cache-dir . && rm -rf /opt/cbtv/build

RUN useradd --create-home --uid 1000 cbtv
USER cbtv
WORKDIR /data
ENV CUDA_VISIBLE_DEVICES=-1 \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["cbtv"]
CMD ["--help"]
