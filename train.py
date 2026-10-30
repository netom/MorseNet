#!/usr/bin/env python3

"""
Training script for morse code decoder using TensorFlow 2.x.

This script trains an LSTM-based neural network with CTC loss to decode
morse code from raw audio data. Uses custom training loop with gradient tape
for maximum control over the training process.
"""

from __future__ import generator_stop

import numpy as np
import pathlib
import random
import tensorflow as tf
import time

from config import *
from model import create_cw_model, ctc_decode

class CTCTrainer:
    """Custom training class for CTC model with checkpoint management."""

    def __init__(self, model, optimizer, checkpoint_dir, log_dir):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Metrics
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.train_ctc_loss_metric = tf.keras.metrics.Mean(name='train_ctc_loss')
        self.train_l2_loss_metric = tf.keras.metrics.Mean(name='train_l2_loss')
        self.train_ler_metric = tf.keras.metrics.Mean(name='train_ler')

        # Checkpoint management
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.model,
            step=tf.Variable(0, dtype=tf.int64),
            epoch=tf.Variable(0, dtype=tf.int64)
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=str(self.checkpoint_dir),
            max_to_keep=CHECKPOINTS_TO_KEEP
        )

        # TensorBoard writer
        self.train_writer = tf.summary.create_file_writer(
            str(self.log_dir / 'train')
        )

        # Restore latest checkpoint if exists
        if self.checkpoint_manager.latest_checkpoint:
            status = self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            checkpoint_variables = dict(tf.train.list_variables(
                self.checkpoint_manager.latest_checkpoint
            ))
            if 'epoch/.ATTRIBUTES/VARIABLE_VALUE' not in checkpoint_variables:
                # Older checkpoints saved batch steps but not completed epochs.
                self.checkpoint.epoch.assign(
                    self.checkpoint.step // NUM_BATCHES_PER_EPOCH
                )
            print(f"Restored from {self.checkpoint_manager.latest_checkpoint}")
            print(f"Starting from step {int(self.checkpoint.step)}")
            print(f"Completed epochs: {int(self.checkpoint.epoch)}")
        else:
            print("Starting training from scratch")

    @tf.function
    def train_step(self, audio, labels, input_length, label_length):
        """
        Single training step with automatic differentiation.

        Args:
            audio: Audio input [batch, timesteps, features]
            labels: Sparse tensor with character labels
            input_length: Length of each audio sequence [batch]
            label_length: Length of each label sequence [batch]

        Returns:
            Tuple of (total_loss, ctc_loss, l2_loss, ler)
        """
        with tf.GradientTape() as tape:
            logits = self.model(audio, training=True)

            loss = tf.reduce_mean(tf.nn.ctc_loss(
                labels=labels,
                logits=logits,
                label_length=label_length,
                logit_length=input_length,
                logits_time_major=False,
                blank_index=0
            ))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        decoded, _ = ctc_decode(
            logits,
            sequence_length=input_length,
            beam_width=1
        )

        ler = tf.reduce_mean(
            tf.edit_distance(
                tf.cast(decoded[0], tf.int32),
                labels
            )
        )

        return loss, ler

    def train_epoch(self, dataset, epoch):
        """
        Train for one epoch.

        Args:
            dataset: tf.data.Dataset with training data
            epoch: Current epoch number

        Returns:
            Dictionary with epoch metrics
        """
        self.train_loss_metric.reset_state()
        self.train_ler_metric.reset_state()

        start_time = time.time()

        for batch_idx, (audio, labels) in enumerate(dataset):
            # Input length is constant for training
            batch_size = tf.shape(audio)[0]
            input_length = tf.fill([batch_size], TIMESTEPS)

            # Label length from sparse tensor
            # Use bincount to count labels per batch element
            batch_indices = labels.indices[:, 0]
            label_length = tf.math.bincount(
                tf.cast(batch_indices, tf.int32),
                minlength=batch_size,
                maxlength=batch_size
            )

            # Train step
            loss, ler = self.train_step(
                audio, labels, input_length, label_length
            )

            # Update metrics
            self.train_loss_metric.update_state(loss)
            self.train_ler_metric.update_state(ler)

            # Increment step counter
            self.checkpoint.step.assign_add(1)

            # Log progress
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch:3d}, Batch {batch_idx:3d}/{NUM_BATCHES_PER_EPOCH}: "
                    f"Loss={loss.numpy():.4f} "
                    f"LER={ler.numpy():.4f}"
                )

        epoch_time = time.time() - start_time

        return {
            'loss': self.train_loss_metric.result().numpy(),
            'ler': self.train_ler_metric.result().numpy(),
            'time': epoch_time
        }

    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        save_path = self.checkpoint_manager.save()
        print(f"Saved checkpoint for epoch {epoch}: {save_path}")
        return save_path

    def log_metrics(self, metrics, epoch):
        """Log metrics to TensorBoard."""
        with self.train_writer.as_default():
            tf.summary.scalar('loss', metrics['loss'], step=epoch)
            tf.summary.scalar('ler', metrics['ler'], step=epoch)
            tf.summary.scalar('epoch_time', metrics['time'], step=epoch)
        self.train_writer.flush()

def get_saved_samples(data_dir):
    """Return matching audio/transcript paths from a saved sample directory."""
    data_dir = pathlib.Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Training data directory does not exist: {data_dir}")

    audio_files = {path.stem: path for path in data_dir.glob('*.npy')}
    transcript_files = {
        path.stem: path for path in data_dir.glob('*.txt')
        if path.name != 'config.txt'
    }

    missing_transcripts = sorted(audio_files.keys() - transcript_files.keys())
    missing_audio = sorted(transcript_files.keys() - audio_files.keys())
    if missing_transcripts or missing_audio:
        details = []
        if missing_transcripts:
            details.append(f"missing transcripts for: {', '.join(missing_transcripts)}")
        if missing_audio:
            details.append(f"missing audio for: {', '.join(missing_audio)}")
        raise ValueError(f"Incomplete training samples in {data_dir}: {'; '.join(details)}")

    samples = [
        (audio_files[stem], transcript_files[stem])
        for stem in sorted(audio_files)
    ]
    if not samples:
        raise ValueError(f"No .npy/.txt training sample pairs found in {data_dir}")

    return samples

def saved_sample_generator(samples):
    """Shuffle and load saved samples continuously, one sample at a time."""
    while True:
        shuffled_samples = list(samples)
        random.shuffle(shuffled_samples)

        for audio_path, transcript_path in shuffled_samples:
            audio = np.load(audio_path, allow_pickle=False)
            if audio.shape != (SEQ_LENGTH_FRAMES,):
                raise ValueError(
                    f"Invalid audio shape in {audio_path}: expected "
                    f"({SEQ_LENGTH_FRAMES},), got {audio.shape}"
                )
            audio = np.reshape(audio.astype(np.float32), (TIMESTEPS, CHUNK))

            characters = transcript_path.read_text(encoding='utf-8').rstrip('\r\n')
            unknown_characters = sorted(set(characters) - MORSE_ORD.keys())
            if unknown_characters:
                raise ValueError(
                    f"Unknown transcript characters in {transcript_path}: "
                    f"{', '.join(repr(character) for character in unknown_characters)}"
                )
            values = np.asarray([MORSE_ORD[character] for character in characters],
                                dtype=np.int32)
            indices = np.reshape(np.arange(len(values), dtype=np.int64), (-1, 1))
            dense_shape = np.asarray([len(values)], dtype=np.int64)
            yield audio, indices, values, dense_shape

def create_dataset_from_saved_samples(data_dir, batch_size, num_batches):
    """Create a shuffled, repeating dataset from saved audio/transcript pairs."""
    samples = get_saved_samples(data_dir)
    print(f"Found {len(samples)} saved training samples in {data_dir}")

    dataset = tf.data.Dataset.from_generator(
        lambda: saved_sample_generator(samples),
        output_signature=(
            tf.TensorSpec(shape=(TIMESTEPS, CHUNK), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(1,), dtype=tf.int64)
        )
    )
    # Construct sparse labels in the dataset graph, not in the Python callback.
    dataset = dataset.map(
        lambda audio, indices, values, dense_shape: (
            audio, tf.SparseTensor(indices, values, dense_shape)
        )
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.take(num_batches)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def main():
    """Main training loop."""

    print("")
    print("Morse Code Decoder Training - TensorFlow 2.x")
    print("")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print(f"Training data dir: {TRAINING_DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches per epoch: {NUM_BATCHES_PER_EPOCH}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Log dir: {LOG_DIR}")
    print("")

    model = create_cw_model()
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    trainer = CTCTrainer(model, optimizer, CHECKPOINT_DIR, LOG_DIR)

    print(f"\nStarting training...")
    print(f"TensorBoard: tensorboard --logdir={LOG_DIR}")
    print("")

    # Training loop
    train_dataset = create_dataset_from_saved_samples(
        TRAINING_DATA_DIR, BATCH_SIZE, NUM_BATCHES_PER_EPOCH
    )
    try:
        for epoch in range(int(trainer.checkpoint.epoch), MAX_EPOCHS):
            print("")
            print(f"Epoch {epoch + 1}/{MAX_EPOCHS}")
            print("")

            # Train
            metrics = trainer.train_epoch(train_dataset, epoch)
            trainer.checkpoint.epoch.assign(epoch + 1)

            # Log to TensorBoard
            trainer.log_metrics(metrics, epoch)

            # Print epoch summary
            print("")
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Loss:     {metrics['loss']:.4f}")
            print(f"  LER:      {metrics['ler']:.4f}")
            print(f"  Time:     {metrics['time']:.2f}s")
            print(f"")

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                trainer.save_checkpoint(epoch + 1)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving final checkpoint...")
        trainer.save_checkpoint(int(trainer.checkpoint.epoch))

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
