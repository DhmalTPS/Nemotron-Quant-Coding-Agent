# SPDX-License-Identifier: Apache-2.0
# Agenthon 2026 Track 1 agent image.
#
# The Python environment is the organizers' shared sandbox (finance-bench-sandbox:latest, built from
# track1-coding-public/docker/sandbox.Dockerfile) plus numba, copied byte-for-byte onto the same
# python:3.13-slim base WITHOUT the build toolchain (gcc/g++/build-essential: ~340 MB, unused at run
# time) and split into ~8 layers of 140-210 MB. Smaller, split layers pull faster at evaluation time
# (pull time counts against the unit clock) and push reliably over slow uplinks.

# ---- stage 1: the official sandbox + numba (graded environment pins numba 0.67.0) -------------
FROM finance-bench-sandbox:latest AS sandbox
RUN pip install --no-cache-dir numba==0.67.0

# ---- stage 2: regroup site-packages into layer-sized trees -------------------------------------
FROM sandbox AS split
RUN set -eux; SP=/usr/local/lib/python3.13/site-packages; \
    grp() { g="$1"; shift; mkdir -p /grp/$g/site-packages; for n in "$@"; do \
        for p in $SP/$n; do [ -e "$p" ] && mv "$p" /grp/$g/site-packages/; done; done; }; \
    grp 1 llvmlite numba; \
    grp 2 _polars_runtime_32 polars; \
    grp 3 pyarrow; \
    grp 4 scipy scipy.libs; \
    grp 5 pandas numpy numpy.libs; \
    grp 6 statsmodels sklearn scikit_learn.libs patsy formulaic; \
    grp 7 plotly matplotlib fontTools PIL pillow.libs kiwisolver contourpy; \
    mkdir -p /grp/8; mv $SP /grp/8/site-packages

# ---- stage 3: runtime image --------------------------------------------------------------------
FROM python:3.13-slim
COPY --from=split /grp/1/ /usr/local/lib/python3.13/
COPY --from=split /grp/2/ /usr/local/lib/python3.13/
COPY --from=split /grp/3/ /usr/local/lib/python3.13/
COPY --from=split /grp/4/ /usr/local/lib/python3.13/
COPY --from=split /grp/5/ /usr/local/lib/python3.13/
COPY --from=split /grp/6/ /usr/local/lib/python3.13/
COPY --from=split /grp/7/ /usr/local/lib/python3.13/
COPY --from=split /grp/8/ /usr/local/lib/python3.13/
COPY --from=sandbox /usr/local/bin/ /usr/local/bin/
COPY --from=sandbox /usr/lib/libta-lib* /usr/lib/

# Platform: non-root uid 65534, read-only root FS, no HOME, 64 MiB noexec /tmp.
ENV HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl \
    NUMBA_CACHE_DIR=/tmp/numba \
    XDG_CACHE_HOME=/tmp/cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/opt/agent

COPY LICENSE /opt/agent/LICENSE
COPY agent /opt/agent/agent
COPY assets /opt/agent/assets
RUN python -m compileall -q /opt/agent/agent \
    && chmod -R a+rX /opt/agent \
    && printf '#!/bin/sh\nexec python -m agent "$@"\n' > /usr/local/bin/solve \
    && chmod 755 /usr/local/bin/solve \
    && mkdir -p /app/output /output \
    && python -c "import agent.cli, numpy, pandas, scipy, statsmodels, sklearn, arch, polars, pyarrow, numba, talib, matplotlib, seaborn, plotly, openpyxl, backtrader, pytest; print('agent image ok')"

LABEL qfbench2.interface_version="2.0"
WORKDIR /tmp
ENTRYPOINT ["python", "-m", "agent"]
CMD ["solve", "--task-dir", "/input", "--out", "/app/output"]
