#!/usr/bin/env python3

"""
Training script for morse code decoder using TensorFlow 2.x.

This script trains an LSTM-based neural network with CTC loss to decode
morse code from raw audio data. Uses custom training loop with gradient tape
for maximum control over the training process.
"""

from __future__ import generator_stop

import tensorflow as tf
import numpy as np
import time
from datetime import datetime
from pathlib import Path

from config import *
import generate_wav_samples as gen
from model import create_cw_model, ctc_decode

# Training configuration
BATCH_SIZE = 100
NUM_BATCHES_PER_EPOCH = 60
MAX_EPOCHS = 1000
CHECKPOINT_DIR = './model_train'
LOG_DIR = f'./logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
L2_LAMBDA = 0.005
GRADIENT_CLIP_NORM = 1.0

class CTCTrainer:
    """Custom training class for CTC model with checkpoint management."""

    def __init__(self, model, optimizer, checkpoint_dir, log_dir):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir = Path(log_dir)
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
            step=tf.Variable(0, dtype=tf.int64)
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=str(self.checkpoint_dir),
            max_to_keep=30
        )

        # TensorBoard writer
        self.train_writer = tf.summary.create_file_writer(
            str(self.log_dir / 'train')
        )

        # Restore latest checkpoint if exists
        if self.checkpoint_manager.latest_checkpoint:
            status = self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"Restored from {self.checkpoint_manager.latest_checkpoint}")
            print(f"Starting from step {int(self.checkpoint.step)}")
        else:
            print("Starting training from scratch")

    @tf.function(reduce_retracing=True)
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

            ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(
                labels=labels,
                logits=logits,
                label_length=label_length,
                logit_length=input_length,
                logits_time_major=False,
                blank_index=NUM_CLASSES-1
            ))

            # L2 regularization (exclude bias terms)
            l2_loss = tf.add_n([
                tf.nn.l2_loss(var)
                for var in self.model.trainable_variables
                if 'bias' not in var.name.lower()
            ])
            l2_loss *= L2_LAMBDA

            # Total loss
            total_loss = ctc_loss + l2_loss

        # Compute gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)

        # Clip gradients by global norm (use smaller value for stability)
        gradients, _ = tf.clip_by_global_norm(gradients, 0.5)

        # Apply gradients
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        # Decode for LER calculation (use greedy decoder for speed)
        decoded, _ = ctc_decode(
            logits,
            sequence_length=input_length,
            use_beam_search=False
        )

        # Calculate Label Error Rate using edit distance
        ler = tf.reduce_mean(
            tf.edit_distance(
                tf.cast(decoded[0], tf.int32),
                labels
            )
        )

        return total_loss, ctc_loss, l2_loss, ler

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
        self.train_ctc_loss_metric.reset_state()
        self.train_l2_loss_metric.reset_state()
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
            total_loss, ctc_loss, l2_loss, ler = self.train_step(
                audio, labels, input_length, label_length
            )

            # Update metrics
            self.train_loss_metric.update_state(total_loss)
            self.train_ctc_loss_metric.update_state(ctc_loss)
            self.train_l2_loss_metric.update_state(l2_loss)
            self.train_ler_metric.update_state(ler)

            # Increment step counter
            self.checkpoint.step.assign_add(1)

            # Log progress
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch:3d}, Batch {batch_idx:3d}/{NUM_BATCHES_PER_EPOCH}: "
                    f"Loss={total_loss.numpy():.4f} "
                    f"(CTC={ctc_loss.numpy():.4f}, L2={l2_loss.numpy():.4f}), "
                    f"LER={ler.numpy():.4f}"
                )

        epoch_time = time.time() - start_time

        return {
            'loss': self.train_loss_metric.result().numpy(),
            'ctc_loss': self.train_ctc_loss_metric.result().numpy(),
            'l2_loss': self.train_l2_loss_metric.result().numpy(),
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
            tf.summary.scalar('ctc_loss', metrics['ctc_loss'], step=epoch)
            tf.summary.scalar('l2_loss', metrics['l2_loss'], step=epoch)
            tf.summary.scalar('ler', metrics['ler'], step=epoch)
            tf.summary.scalar('epoch_time', metrics['time'], step=epoch)


def create_dataset(batch_size, num_batches):
    """
    Create tf.data.Dataset from generator.

    Args:
        batch_size: Number of samples per batch
        num_batches: Number of batches per epoch

    Returns:
        tf.data.Dataset
    """

    def generator_wrapper():
        """Wrapper for generator to match output signature."""
        for audio, label_indices, label_values, label_shape in gen.seq_generator(
            SEQ_LENGTH, FRAMERATE, CHUNK
        ):
            #print("GENWRAP", len(label_indices), len(label_values), label_shape)
            # Convert to sparse tensor immediately with explicit int32 casting
            # (ctc_loss expects int32 for labels)
            sparse_label = tf.SparseTensor(
                indices=tf.cast(label_indices, tf.int64),
                values=tf.cast(label_values, tf.int32),
                dense_shape=tf.cast(label_shape, tf.int64)
            )
            #print("GENWRAP", len(audio), sparse_label.shape)
            yield audio, sparse_label

    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        generator_wrapper,
        output_signature=(
            tf.TensorSpec(shape=(TIMESTEPS, CHUNK), dtype=tf.float32),
            tf.SparseTensorSpec(shape=(None,), dtype=tf.int32)
        )
    )

    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Limit to num_batches
    dataset = dataset.take(num_batches)

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def main():
    """Main training loop."""

    print("="*70)
    print("Morse Code Decoder Training - TensorFlow 2.x")
    print("="*70)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches per epoch: {NUM_BATCHES_PER_EPOCH}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Log dir: {LOG_DIR}")
    print("="*70)

    # Create model
    print("\nCreating model...")
    model = create_cw_model()

    print("\nModel architecture:")
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Create trainer
    trainer = CTCTrainer(model, optimizer, CHECKPOINT_DIR, LOG_DIR)

    print(f"\nStarting training...")
    print(f"TensorBoard: tensorboard --logdir={LOG_DIR}")
    print("="*70)

    # Training loop
    try:
        for epoch in range(MAX_EPOCHS):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{MAX_EPOCHS}")
            print(f"{'='*70}")

            # Create dataset for this epoch
            train_dataset = create_dataset(BATCH_SIZE, NUM_BATCHES_PER_EPOCH)

            # Train
            metrics = trainer.train_epoch(train_dataset, epoch)

            # Log to TensorBoard
            trainer.log_metrics(metrics, epoch)

            # Print epoch summary
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Loss:     {metrics['loss']:.4f}")
            print(f"  CTC Loss: {metrics['ctc_loss']:.4f}")
            print(f"  L2 Loss:  {metrics['l2_loss']:.4f}")
            print(f"  LER:      {metrics['ler']:.4f}")
            print(f"  Time:     {metrics['time']:.2f}s")
            print(f"{'='*70}")

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                trainer.save_checkpoint(epoch + 1)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving final checkpoint...")
        trainer.save_checkpoint(epoch + 1)

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
