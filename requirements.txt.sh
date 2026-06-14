#!/usr/bin/env bash

: ${WITH_CUDA:=""}

if [ -n "$WITH_CUDA" ]; then
    AND_MAYBE_CUDA="[and-cuda]"
else
    AND_MAYBE_CUDA=""
fi

cat <<EOF
numpy==2.4.6
scipy==1.17.1
setuptools==81.0.0
tensorboard==2.20.0
tensorflow${AND_MAYBE_CUDA}==2.21.0
EOF
