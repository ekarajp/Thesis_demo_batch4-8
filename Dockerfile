# rc-fragility-poc — locked Python 3.10 runtime for Linux.
#
# This image pins the exact scientific environment the thesis results were
# produced under: 64-bit CPython 3.10 + openseespy 3.5.1.3. The bundled
# wheelhouse/ is Windows-only and is intentionally NOT used here; dependencies
# come from PyPI, with the Windows OpenSeesPy backend (openseespywin) replaced
# by its Linux counterpart (openseespylinux, resolved automatically by openseespy).
FROM python:3.10-slim

# Runtime shared libraries for the prebuilt OpenSees binary (Fortran/OpenMP
# runtime) + ca-certificates for pip. No compiler toolchain is needed because
# every dependency ships a binary manylinux wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgfortran5 \
        libgomp1 \
        libquadmath0 \
        libblas3 \
        liblapack3 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/app/src \
    MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---- dependency layer (cached independently of source) ----
COPY requirements-lock.txt /app/
RUN pip install --no-cache-dir setuptools==75.8.2 wheel==0.45.1 \
    && grep -viE '^openseespywin' /app/requirements-lock.txt > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt \
    && rm /tmp/reqs.txt

# ---- application + committed data (ground motion, sqlite schema) ----
COPY . /app

# package_manifest.json is a transport-integrity check for the distributable
# Windows ZIP (it records file sizes + the win_amd64 wheelhouse). It is stale
# relative to the git source and not applicable here. prepare_portable_runtime.py
# skips that manifest check once a prepared.json marker exists and instead still
# enforces the meaningful per-file checks: ground-motion sha256 vs the SQLite
# catalog and the frozen active-IDA controller sha256 vs config/poc.json. Seed it.
RUN mkdir -p /app/runtime \
    && printf '{"schema":"portable-runtime-prepared-v1","package_root":"/app","prepared_utc":"docker-build"}' \
       > /app/runtime/prepared.json

# Verify the locked environment imports cleanly.
RUN python -c "import fragility_poc, numpy, scipy, sklearn, openseespy.opensees; print('RC Fragility environment import: PASS')"

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8765

# prepare -> preflight -> dashboard (override CMD for one-off CLI runs)
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "/app/server_dashboard.py", "--host", "0.0.0.0", "--port", "8765"]
