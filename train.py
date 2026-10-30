#!/usr/bin/env python3

"""
Training script for morse code decoder using TensorFlow 2.x.

This script trains an LSTM-based neural network with CTC loss to decode
morse code from raw audio data. Uses custom training loop with gradient tape
for maximum control over the training process.
"""

from __future__ import generator_stop

import pathlib
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

# Size of the header np.save writes in front of the data of a 1-D float32 array
NPY_HEADER_BYTES = 128

def create_dataset_from_saved_samples(data_dir, batch_size, num_batches):
    """
    Create a shuffled, repeating dataset from saved .npy audio / .txt
    transcript.
    """
    char_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(MORSE_CHR), tf.range(NUM_CLASSES, dtype=tf.int32)
        ),
        default_value=-1
    )
    audio_bytes = 4 * SEQ_LENGTH_FRAMES

    def load_sample(audio_path):
        raw = tf.io.read_file(audio_path)
        tf.debugging.Assert(
            tf.strings.length(raw) == NPY_HEADER_BYTES + audio_bytes,
            ["Unexpected .npy size (expected float32 samples only):", audio_path]
        )
        audio = tf.io.decode_raw(
            tf.strings.substr(raw, NPY_HEADER_BYTES, audio_bytes), tf.float32
        )
        audio = tf.reshape(audio, (TIMESTEPS, CHUNK))

        text = tf.io.read_file(tf.strings.regex_replace(audio_path, r'\.npy$', '.txt'))
        text = tf.strings.regex_replace(text, r'[\r\n]+$', '')
        labels = char_table.lookup(tf.strings.bytes_split(text))
        tf.debugging.Assert(
            tf.reduce_all(labels >= 0),
            ["Unknown character in transcript of", audio_path]
        )
        return audio, labels

    return (
        tf.data.Dataset.list_files(f'{data_dir}/*.npy', shuffle=True)
        .repeat()
        .map(load_sample, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        .ragged_batch(batch_size, drop_remainder=True)
        .map(lambda audio, labels: (audio, labels.to_sparse()))
        .take(num_batches)
        .prefetch(tf.data.AUTOTUNE)
    )

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
