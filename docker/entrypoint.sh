#!/bin/sh
# docker/entrypoint.sh  (mounted from /workspace/docker/entrypoint.sh; must be executable)
#
# Stage 4 (plan §5.8): editable-install the repository into the `exp` conda environment at
# container start, then hand control to the service's own command.
#
#   --no-deps               conda owns the scientific environment; pip must not resolve it (R14).
#   --no-build-isolation    use the setuptools/wheel already present in `exp`; never download (R17).
#   --disable-pip-version-check
#                           no PyPI request at container start.
#
# The install is unconditional: a guard such as `python -c "import noisyvis"` would pass after a
# pyproject.toml change and leave the packaging metadata stale.
#
# `set -eu` makes a failed install fatal *before* exec, so the service never starts with a broken
# import state; the container dies and visibly restarts instead.
set -eu
micromamba run -n exp python -m pip install \
    -e /workspace --no-deps --no-build-isolation --quiet --disable-pip-version-check
exec micromamba run -n exp "$@"
