#!/usr/bin/env python3

"""
Morse code decoder model using TensorFlow 2.x / Keras API.

This module provides the neural network architecture for decoding morse code
from raw audio using LSTM layers and CTC (Connectionist Temporal Classification) loss.
"""

import tensorflow as tf
from config import *


def create_cw_model(
    max_timesteps=TIMESTEPS,
    num_features=CHUNK,
    recurrent_layer_depth=3,
    recurrent_layer_width=128,
    num_classes=NUM_CLASSES
):
    """
    Creates a Keras model for morse code decoding using CTC.

    Architecture:
    - Input: [batch, timesteps, features] - raw audio chunks
    - LSTM layers with layer normalization (replaces LayerNormBasicLSTMCell from TF 1.x)
    - Dense output layer projecting to num_classes
    - CTC loss applied separately in training loop

    Args:
        max_timesteps: Maximum number of timesteps (None for variable length)
        num_features: Number of features per timestep (chunk size)
        recurrent_layer_depth: Number of LSTM layers (standardized to 3)
        recurrent_layer_width: Number of units in each LSTM layer
        num_classes: Number of output classes (morse characters)

    Returns:
        model: tf.keras.Model instance outputting logits [batch, timesteps, num_classes]
    """

    # Input layer - audio chunks
    input_audio = tf.keras.Input(
        shape=(max_timesteps, num_features),
        name='audio_input',
        dtype=tf.float32
    )

    # Recurrent layers with layer normalization
    # This replaces tf.contrib.rnn.LayerNormBasicLSTMCell from TF 1.x
    x = input_audio
    for i in range(recurrent_layer_depth):
        x = tf.keras.layers.LSTM(
            recurrent_layer_width,
            return_sequences=True,  # Return full sequence for CTC
            activation='tanh',  # Use tanh instead of relu for stability
            recurrent_activation='sigmoid',
            use_bias=True,
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
            recurrent_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
            dropout=0.0,  # Disable dropout for now to avoid instability
            recurrent_dropout=0.0,
            name=f'lstm_{i}'
        )(x)

        # Layer normalization (replacement for LayerNormBasicLSTMCell layer_norm)
        x = tf.keras.layers.LayerNormalization(
            epsilon=1e-5,
            name=f'layer_norm_{i}'
        )(x)

    # Output dense layer - project to number of classes
    # Linear activation for CTC (no softmax needed)
    # Use small random initialization to prevent large initial logits
    logits = tf.keras.layers.Dense(
        num_classes,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=tf.keras.initializers.Zeros(),
        name='output_dense'
    )(x)

    model = tf.keras.Model(
        inputs=input_audio,
        outputs=logits,
        name='cw_decoder'
    )

    return model


def ctc_loss_fn(labels, logits, input_length, label_length, blank_index=None):
    """
    Compute CTC loss for training.

    Args:
        labels: Sparse tensor with label sequences [batch, max_label_length]
        logits: Model outputs [batch, timesteps, num_classes]
        input_length: Actual length of each input sequence [batch]
        label_length: Actual length of each label sequence [batch]
        blank_index: Index for blank label (default: NUM_CLASSES - 1)

    Returns:
        Mean CTC loss across batch
    """
    if blank_index is None:
        blank_index = NUM_CLASSES - 1

    # Clip logits to prevent numerical instability
    logits = tf.clip_by_value(logits, -50.0, 50.0)

    # CTC expects time-major format [timesteps, batch, num_classes]
    logits_transposed = tf.transpose(logits, [1, 0, 2])

    # Compute CTC loss with numerical stability
    loss = tf.nn.ctc_loss(
        labels=labels,
        logits=logits_transposed,
        label_length=label_length,
        logit_length=input_length,
        logits_time_major=True,
        blank_index=blank_index
    )

    # Clip loss to prevent NaN propagation
    loss = tf.clip_by_value(loss, 0.0, 1e10)

    return tf.reduce_mean(loss)


def ctc_decode(logits, sequence_length, beam_width=100, use_beam_search=True):
    """
    Decode CTC outputs to character sequences.

    Args:
        logits: Model outputs [batch, timesteps, num_classes]
        sequence_length: Actual length of each sequence [batch]
        beam_width: Beam width for beam search decoder
        use_beam_search: If True use beam search, else use greedy decoder

    Returns:
        decoded: List of sparse tensors with decoded sequences
        log_prob: Log probabilities of decoded sequences
    """
    # CTC decoders expect time-major format
    logits_transposed = tf.transpose(logits, [1, 0, 2])

    if use_beam_search:
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(
            logits_transposed,
            sequence_length=sequence_length,
            beam_width=beam_width
        )
    else:
        decoded, log_prob = tf.nn.ctc_greedy_decoder(
            logits_transposed,
            sequence_length=sequence_length
        )

    return decoded, log_prob


def decoded_to_text(decoded_sparse_tensor, character_set=MORSE_CHR):
    """
    Convert decoded sparse tensor to readable text.

    Args:
        decoded_sparse_tensor: Sparse tensor from CTC decoder
        character_set: List of characters in order

    Returns:
        List of decoded strings (one per batch element)
    """
    # Convert sparse to dense
    decoded_dense = tf.sparse.to_dense(
        decoded_sparse_tensor,
        default_value=-1
    ).numpy()

    # Convert indices to characters
    result = []
    for sequence in decoded_dense:
        text = ''.join([
            character_set[idx] for idx in sequence
            if idx >= 0 and idx < len(character_set)
        ])
        result.append(text)

    return result


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")

    model = create_cw_model(
        max_timesteps=TIMESTEPS,
        num_features=CHUNK,
        recurrent_layer_depth=3,
        recurrent_layer_width=128,
        num_classes=NUM_CLASSES
    )

    model.summary()

    print("\nModel created successfully!")
    print(f"Input shape: [batch, {TIMESTEPS}, {CHUNK}]")
    print(f"Output shape: [batch, {TIMESTEPS}, {NUM_CLASSES}]")
