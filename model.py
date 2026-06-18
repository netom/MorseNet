#!/usr/bin/env python3

import tensorflow as tf
from config import *

# TODO: dropout
def create_cw_model(
    input_widths=[128],
    recurrent_widths=[128,128,128],
    output_widths=[NUM_CLASSES],
):
    layers = []

    for i, width in enumerate(input_widths):
        layers.append(tf.keras.layers.Dense(
            width,
            kernel_initializer = tf.keras.initializers.Orthogonal(1.0),
            bias_initializer = tf.keras.initializers.Zeros(),
            activation=None,
            name=f'input_{i}'
        ))

    for i, width in enumerate(recurrent_widths):
        layers.append(tf.keras.layers.LSTM(
            width,
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True,
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
            recurrent_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
            dropout=0.0,
            recurrent_dropout=0.0,
            name=f'lstm_{i}'
        ))

    for i, width in enumerate(output_widths):
        layers.append(tf.keras.layers.Dense(
            width,
            activation=None,
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
            bias_initializer=tf.keras.initializers.Zeros(),
            name=f'output_{i}'
        ))

    model = tf.keras.Sequential(layers)
    model.run_eagerly=True

    return model

def ctc_decode(logits, sequence_length, beam_width=100):
    """
    Decode CTC outputs to character sequences.

    tf.nn.ctc_beam_search_decoder has no blank_index parameter - it always
    treats index num_classes-1 as blank. To support blank at index 0
    (matching tf.nn.ctc_loss blank_index=0), the class axis is rotated before
    decoding (blank moves from front to back) and decoded indices are shifted
    back by +1 afterward.

    Args:
        logits: Model outputs [batch, timesteps, num_classes]
        sequence_length: Actual length of each sequence [batch]
        beam_width: Beam width for beam search decoder

    Returns:
        decoded: List of sparse tensors with decoded sequences
        log_prob: Log probabilities of decoded sequences
    """
    # CTC decoders expect time-major format
    logits_transposed = tf.transpose(logits, [1, 0, 2])

    # Rotate class axis: blank (index 0) moves to last position so the
    # decoder's fixed blank=num_classes-1 assumption is satisfied.
    logits_rotated = tf.concat([
        logits_transposed[:, :, 1:],   # non-blank classes shifted left
        logits_transposed[:, :, :1],   # blank moved to last
    ], axis=-1)

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(
        logits_rotated,
        sequence_length=sequence_length,
        beam_width=beam_width,
        top_paths=1
    )

    # Undo the rotation: indices are 0-based over the shifted alphabet, so
    # add 1 to recover the original MORSE_CHR indices.
    decoded = [
        tf.SparseTensor(
            indices=d.indices,
            values=d.values + 1,
            dense_shape=d.dense_shape,
        )
        for d in decoded
    ]

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
