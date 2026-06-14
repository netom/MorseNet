#!/usr/bin/env bash

# $PYTHON can be set to a preferred python interpreter.
: ${PYTHON:=""}

# Virtual environment directory
: ${VENV:=".venv"}

# If $WITH_CUDA is non empty (e.g. "y"), tensorflow is
# installed with CUDA support.
: ${WITH_CUDA:=""}

# To avoid complications, refuse to run in an active venv.
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Already in an active virtual environment, aborting."
    return 1 &> /dev/null || exit 1
fi

# Try to find a python or python3 interpreter.
if [ -z "$PYTHON" ] && command -v python3 > /dev/null; then
    PYTHON=python3
fi

if [ -z "$PYTHON" ] && command -v python > /dev/null; then
    PYTHON=python
fi

if [ -z "$PYTHON" ]; then
    echo "Neither python3 nor python could be found."
    return 1 &> /dev/null || exit 1
fi

if [ -d "$VENV" ]; then
    echo "Using existing virtual environment in $VENV"
else
    $PYTHON -m venv "$VENV"
fi

. .venv/bin/activate

pip install -r <( . requirements.txt.sh )

_OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

export LD_LIBRARY_PATH=$(
    (   tr : "\n" <<< "$LD_LIBRARY_PATH"
        find "$VIRTUAL_ENV" -name "*.so*" | grep nvidia | xargs dirname
    ) | sort -u | grep '[[:graph:]]' | paste -d ":" -s -
)


if [ -z "$WITH_CUDA" ]; then
    export CUDA_VISIBLE_DEVICES=""
fi

eval "_old_$(declare -f deactivate)"
deactivate () {
    _old_deactivate
    LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
    if [ -z "$WITH_CUDA" ]; then
        unset CUDA_VISIBLE_DEVICES
    fi
}
