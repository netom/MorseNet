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

    return train_inputs_, train_seq_len_, train_targets_, raw_targets_

def cw_model(features, labels, mode, params):
    ####################################################################
    # INPUT
    #
    # -VVV- [params['max_timesteps'], params['batch_size'], params['num_features']]

    # Has size [params['max_timesteps'], params['batch_size'], params['num_features']].
    # Note chat params['num_features'] is the size of the audio data chunk processed
    # at each step, which is the number of input features.
    seq_len=tf.constant(params['max_timesteps'], dtype=tf.int32, shape=[params['batch_size']])

    I = features

    # labels must be a SparseTensor required by ctc_loss op.

    ####################################################################
    # INPUT DENSE BAND
    #
    # -^^^- [params['max_timesteps'], params['batch_size'], params['num_features']]
    #I = tf.reshape(I, [-1, params['num_features']])
    # -VVV- [params['max_timesteps'] * params['batch_size'], params['num_features']]


    #I = tf.layers.dense(
    #    I,
    #    256,
    #    kernel_initializer = tf.orthogonal_initializer(1.0),
    #    bias_initializer = tf.zeros_initializer(),
    #    activation=tf.nn.relu
    #)

    ####################################################################
    # RECURRENT BAND
    #
    # -^^^- [params['max_timesteps'] * params['batch_size'], 128]
    #I = tf.reshape(I, [params['max_timesteps'], params['batch_size'], 256])
    # -VVV- [params['max_timesteps'], params['batch_size'], 128]

    with tf.variable_scope("", initializer=tf.orthogonal_initializer(0.9)):
        lstmbfc = tf.contrib.rnn.LSTMBlockFusedCell(128) # Creates a factory
        I, _ = lstmbfc(I, initial_state=None, dtype=tf.float32) # Actually retrieves the output. Clever.

    ####################################################################
    # OUTPUT DENSE BAND
    #
    # -^^^- [params['max_timesteps'], params['batch_size'], 128]
    I = tf.reshape(I, [params['max_timesteps'] * params['batch_size'], 128])
    # -VVV- [params['max_timesteps'] * params['batch_size'], 128]

    I = tf.layers.dense(
        I,
        NUM_CLASSES,
        kernel_initializer = tf.orthogonal_initializer(1.0),
        bias_initializer = tf.zeros_initializer(),
        activation=tf.nn.relu
    )

    ####################################################################
    # OUTPUT
    #
    # -^^^- [params['max_timesteps'] * params['batch_size'], NUM_CLASSES]
    I = tf.reshape(I, [params['max_timesteps'], params['batch_size'], NUM_CLASSES])
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

    metrics = {'ler': ler}
    tf.summary.scalar('ler', ler)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(args):
    print("*** LOADING DATA ***")

    num_epochs = 10000
    train_batch_size = 500
    valid_batch_size = 500
    num_batches_per_epoch = 1
    num_examples = num_batches_per_epoch * train_batch_size

    features, seq_len, labels, valid_raw_targets = load_batch(20, valid_batch_size)

    tfconfig = tf.ConfigProto(
        device_count = {
            'GPU': 0,
            #'CPU': 8
        },
        #intra_op_parallelism_threads = 16,
        #inter_op_parallelism_threads = 16,
        log_device_placement = False,
        #allow_soft_placement = True
    )

    estimator = tf.estimator.Estimator(
        model_fn=cw_model,
        model_dir='./model_dir',
        params={
            'max_timesteps': MAX_TIMESTEPS,
            'batch_size': train_batch_size,
            'num_features': CHUNK
        }
    )

    # Train the Model.
    estimator.train(
        input_fn=lambda:tf.data.Dataset.from_tensors((features, labels)).repeat(),
        steps=10000
    )

    # Evaluate the model.
    #eval_result = estimator.evaluate(
    #    input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size)
    #)

    #print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

tf.logging.set_verbosity(tf.logging.INFO)
tf.app.run(main)
