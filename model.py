#!/usr/bin/env python3

import tensorflow as tf
from config import *

def create_cw_model(
    max_timesteps=TIMESTEPS,
    num_features=CHUNK,
    num_classes=NUM_CLASSES,
    input_layer_depth=0,
    input_layer_width=CHUNK,
    recurrent_layer_depth=2,
    recurrent_layer_width=128,
    output_layer_depth=1,
    output_layer_width=128
):
    # Input dense layers
    input_dense = []
    for i in range(input_layer_depth):
        input_dense.append(tf.keras.layers.Dense(
            input_layer_width,
            kernel_initializer = tf.keras.initializers.Orthogonal(1.0),
            bias_initializer = tf.keras.initializers.Zeros(),
            activation=None,
            name=f'input_dense_{i}'
        ))
        # TODO: dropout

    # Recurrent layers with layer normalization
    # This replaces tf.contrib.rnn.LayerNormBasicLSTMCell from TF 1.x
    recurrent = []
    for i in range(recurrent_layer_depth):
        recurrent.append(tf.keras.layers.LSTM(
            recurrent_layer_width,
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

        recurrent.append(tf.keras.layers.LayerNormalization(
            name=f'layer_norm_{i}'
        ))

        # TODO: dropout?

    output_dense = []
    for i in range(recurrent_layer_depth):
        output_dense.append(tf.keras.layers.Dense(
            num_classes,
            activation=None,
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
            bias_initializer=tf.keras.initializers.Zeros(),
            name=f'output_dense_{i}'
        ))

    model = tf.keras.Sequential(input_dense + recurrent + output_dense)

    return model

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
