#!/usr/bin/env bash

PYTHON=""

if command -v python3 > /dev/null; then
    PYTHON=python3
fi

if command -v python > /dev/null; then
    PYTHON=python
fi

if [ -n "$PYTHON" ]; then
    python -m venv .venv

    . .venv/bin/activate

    pip install -r requirements.txt
else
    echo "Couldn't run python3 nor python."
fi

