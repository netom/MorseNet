#!/usr/bin/env python3

import time

import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np
import sys
import matplotlib.pyplot as plt

from config import *

def load_batch(batch_id, batch_size):
    target_filename_tpl = 'training_set/%04d/%03d.txt'
    audio_filename_tpl  = 'training_set/%04d/%03d.wav'

    train_inputs_  = []
    train_targets_ = []
    raw_targets_   = []

    # Files must be of the same length in one batch
    for i in range(batch_size):
        audio_filename = audio_filename_tpl % (batch_id, i)

        fs, audio = wav.read(audio_filename)

        time_steps = len(audio)//CHUNK
        truncated_autio_length = time_steps * CHUNK

        # Input shape is [num_batches, time_steps, CHUNK (features)]
        inputs = np.reshape(audio[:truncated_autio_length],  (time_steps, CHUNK))
        inputs = (inputs - np.mean(inputs)) / np.std(inputs) # Normalization
        #inputs = np.fft.rfft(inputs)[:,16:23] # FFT
        #plt.imshow(np.absolute(inputs))
        #plt.show()

        train_inputs_.append(inputs)
        sys.stdout.write("Loading batch %d: %d... \r" % (batch_id, i))

    train_inputs_  = np.asarray(train_inputs_, dtype=np.float32).transpose((1,0,2))
    train_seq_len_ = np.asarray([time_steps]*batch_size, dtype=np.int32)

    # Read targets
    tt_indices  = []
    tt_values   = []
    max_target_len = 0
    for i in range(batch_size):
        target_filename = target_filename_tpl % (batch_id, i)

        with open(target_filename, 'r') as f:
            targets = list(map(lambda x: x[0], f.readlines()))

        raw_targets_.append(''.join(targets))

        # Transform char into index
        targets = np.asarray([MORSE_CHR.index(x) for x in targets])
        tlen = len(targets)
        if  tlen > max_target_len:
            max_target_len = tlen

        # Creating sparse representation to feed the placeholder
        for j, value in enumerate(targets):
            tt_indices.append([i,j])
            tt_values.append(value)

    # Build a sparse matrix for training required by the ctc loss function
    train_targets_ = tf.SparseTensorValue(
        tt_indices,
        np.asarray(tt_values, dtype=np.int32),
        (batch_size, max_target_len)
    )

    return train_inputs_, train_targets_

def cw_model(features, labels, mode, params):
    ####################################################################
    # INPUT
    #
    # -VVV- [params['batch_size'], params['max_timesteps'], params['num_features']]

    # Has size params['batch_size'], [params['max_timesteps'], params['num_features']].
    # Note chat params['num_features'] is the size of the audio data chunk processed
    # at each step, which is the number of input features.
    seq_len=tf.constant(params['max_timesteps'], dtype=tf.int32, shape=[params['batch_size']])

    I = features

    # labels must be a SparseTensor required by ctc_loss op.

    ####################################################################
    # INPUT DENSE BAND
    #
    # -^^^- [params['batch_size'], params['max_timesteps'], params['num_features']]
    I = tf.reshape(I, [params['batch_size'] * params['max_timesteps'], params['num_features']])
    # -VVV- [params['batch_size'] * params['max_timesteps'], params['num_features']]

    I = tf.layers.dense(
        I,
        128,
        kernel_initializer = tf.orthogonal_initializer(0.9),
        bias_initializer = tf.zeros_initializer(),
        activation=tf.nn.relu,
        name="inputDense0"
    )
    I = tf.layers.dense(
        I,
        128,
        kernel_initializer = tf.orthogonal_initializer(0.9),
        bias_initializer = tf.zeros_initializer(),
        activation=tf.nn.relu,
        name="inputDense1"
    )

    ####################################################################
    # RECURRENT BAND
    #
    # -^^^- [params['batch_size'] * params['max_timesteps'], 128]
    I = tf.reshape(I, [params['batch_size'], params['max_timesteps'], 128])
    # -VVV- [params['batch_size'], params['max_timesteps'], 128]

    with tf.variable_scope("", initializer=tf.orthogonal_initializer(0.2)):
        #cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
        #)
        cell = tf.nn.rnn_cell.BasicRNNCell(
            128,
            activation=tf.nn.tanh,
            name="recurrent0"
        )
        I, _ = tf.nn.dynamic_rnn(
            cell,
            I,
            sequence_length=seq_len,
            dtype=tf.float32,
            time_major=True
        )

    ####################################################################
    # OUTPUT DENSE BAND
    #
    # -^^^- [params['batch_size'], params['max_timesteps'], 128]
    I = tf.reshape(I, [params['batch_size'] * params['max_timesteps'], 128])
    # -VVV- [params['batch_size'] * params['max_timesteps'], 128]

    I = tf.layers.dense(
        I,
        128,
        kernel_initializer = tf.orthogonal_initializer(0.9),
        bias_initializer = tf.zeros_initializer(),
        activation=tf.nn.relu,
        name="outputDense0"
    )
    I = tf.layers.dense(
        I,
        NUM_CLASSES,
        kernel_initializer = tf.orthogonal_initializer(0.9),
        bias_initializer = tf.zeros_initializer(),
        activation=tf.nn.relu,
        name="outputDense1"
    )

    ####################################################################
    # OUTPUT
    #
    # -^^^- [params['batch_size'] * params['max_timesteps'], NUM_CLASSES]
    I = tf.reshape(I, [params['batch_size'], params['max_timesteps'], NUM_CLASSES])
    I = tf.transpose(I, (1, 0, 2))
    # -VVV- [params['max_timesteps'], params['batch_size'], NUM_CLASSES]


    # ctc_loss is by default time major
    ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(labels, I, seq_len))
    tf.summary.scalar('ctc_loss', ctc_loss)

    # Regularization
    lambda_l2_reg = 0.005
    reg_loss = lambda_l2_reg * tf.reduce_sum([ tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if not ("bias" in tf_var.name) ])
    tf.summary.scalar('reg_loss', reg_loss)

    loss = ctc_loss + reg_loss
    tf.summary.scalar('loss', loss)

    # Old learning rate = 0.0002
    # Treshold = 2.0 step clipping (gradient clipping?)
    #optimizer = tf.train.AdamOptimizer(0.01, 0.9, 0.999, 0.1).minimize(loss)
    optimizer = tf.train.AdamOptimizer()
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs, tf.train.get_global_step())

    decoded, log_prob = tf.nn.ctc_greedy_decoder(I, seq_len)
    #decoded, log_prob = tf.nn.ctc_beam_search_decoder(I, seq_len, beam_width=10)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(
        tf.edit_distance(tf.cast(decoded[0], tf.int32), labels)
    )

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'decoded': decoded,
            'log_prob': log_prob
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    metrics = {
        'ler': (ler, tf.no_op())
    }

    tf.summary.scalar('ler', ler)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op,
        scaffold=tf.train.Scaffold(
            saver=tf.train.Saver(restore_sequentially=True)
        )
    )


def main(args):
    print("*** LOADING DATA ***")

    num_epochs = 100000
    train_batch_size = 100
    valid_batch_size = 100
    num_batches_per_epoch = 20
    num_examples = num_batches_per_epoch * train_batch_size

    labels_i = []
    labels_v = []
    labels_3rd = 0
    features = []
    for i in range(num_batches_per_epoch):
        features_, labels_ = load_batch(i, train_batch_size)
        features.append(features_)
        if labels_.dense_shape[1] > labels_3rd:
            labels_3rd = labels_.dense_shape[1]
        for ind, val in zip(labels_.indices, labels_.values):
            labels_i.append([i] + ind)
            labels_v.append(val)
    features = np.asarray(features)
    labels = tf.SparseTensorValue(labels_i, labels_v, (num_batches_per_epoch, train_batch_size, labels_3rd))

    valid_features, valid_labels = load_batch(20, valid_batch_size)

    estimator = tf.estimator.Estimator(
        model_fn=cw_model,
        model_dir='./model_dir',
        params={
            'max_timesteps': MAX_TIMESTEPS,
            'batch_size': train_batch_size,
            'num_features': CHUNK
        }
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda:tf.data.Dataset.from_tensor_slices((features,labels)).repeat(),
        max_steps=100000
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn = lambda:tf.data.Dataset.from_tensors((valid_features,valid_labels)).repeat(),
        steps=1,
        throttle_secs=1800,
        start_delay_secs=1800,
    )

    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )

tf.logging.set_verbosity(tf.logging.INFO)
tf.app.run(main)
