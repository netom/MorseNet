#!/usr/bin/env python3

"""Export a training checkpoint as a TensorFlow SavedModel."""

import argparse
from pathlib import Path

import tensorflow as tf

from model import build_model

from config.audio import *
from config.train import *

def main():
    parser = argparse.ArgumentParser(
        description="Export a training checkpoint as a TensorFlow SavedModel."
    )
    parser.add_argument(
        "model_path",
        help="Full path of the directory where the SavedModel will be written.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=CHECKPOINT_DIR,
        help=f"Directory containing training checkpoints (default: {CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint prefix to export instead of the latest checkpoint.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint or tf.train.latest_checkpoint(args.checkpoint_dir)
    if checkpoint is None:
        parser.error(f"No checkpoint found in {args.checkpoint_dir}")

    # Keep the exported signature flexible for audio sequences of varying lengths.
    model = build_model()
    model(tf.keras.Input(shape=(None, CHUNK)))

    status = tf.train.Checkpoint(model=model).restore(checkpoint)
    status.expect_partial()

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.export(str(model_path))
    print(f"Exported {checkpoint} to {model_path}")


if __name__ == "__main__":
    main()
