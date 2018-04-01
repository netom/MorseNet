import time

import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np

from config import *

# Hyper-parameters
num_epochs = 200
num_hidden = 128
num_units = 128
num_layers = 1
initial_learning_rate = 1e-2
momentum = 0.8

num_examples = 35
batch_size = num_examples
num_batches_per_epoch = int(num_examples/batch_size)

# Load the data

target_filename_tpl = 'training_set/%04d/%03d.txt'
audio_filename_tpl  = 'training_set/%04d/%03d.wav'

train_inputs  = []
train_targets = []
raw_targets   = []

# Files must be of the same length in one batch
for i in range(batch_size):
    audio_filename = audio_filename_tpl % (0, i)

    fs, audio = wav.read(audio_filename)

    time_steps = len(audio)//CHUNK
    truncated_autio_length = time_steps * CHUNK

    # Input shape is [num_batches, time_steps, CHUNK (features)]
    inputs = np.reshape(audio[:truncated_autio_length],  (time_steps, CHUNK))
    inputs = (inputs - np.mean(inputs)) / np.std(inputs)

    train_inputs.append(inputs)

train_inputs=np.asarray(train_inputs, dtype=np.float32)

# Read targets
tt_indices = []
tt_values  = []
tt_seq_len = []
max_target_len = 0
for i in range(batch_size):
    target_filename = target_filename_tpl % (0, i)

    with open(target_filename, 'r') as f:
        targets = list(map(lambda x: x[0], f.readlines()))

    raw_targets.append(''.join(targets))

    # Transform char into index
    targets = np.asarray([MORSE_CHR.index(x) for x in targets])
    tlen = len(targets)
    if  tlen > max_target_len:
        max_target_len = tlen

    # Oh FUCK YOU very much dear TesorFlow documentation.
    # What the fuck are these mysterious sequences???
    #tt_seq_len.append(tlen+10)
    tt_seq_len.append(time_steps)

    # Creating sparse representation to feed the placeholder
    for j, value in enumerate(targets):
        tt_indices.append([i,j])
        tt_values.append(value)

tt_seq_len = np.asarray(tt_seq_len, dtype=np.int32)

# Build a sparse matrix for training required by the ctc loss function
train_targets = tf.SparseTensorValue(
    tt_indices,
    np.asarray(tt_values, dtype=np.int32),
    (batch_size, max_target_len) # TODO: maximal string length?
)

# Make our validation set to be the 0th one
val_inputs, val_targets, val_seq_len = train_inputs, train_targets, tt_seq_len


# THE MAIN CODE!


graph = tf.Graph()
with graph.as_default():
    # Has size [batch_size, max_stepsize, CHUNK], but the
    # batch_size and max_stepsize can vary along each step
    # Note chat CHUNK is the size of the audio data chunk processed
    # at each step, which is the number of input features.
    inputs = tf.placeholder(tf.float32, [None, None, CHUNK])

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [batch_size])

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell 
    cells = []
    for _ in range(num_layers):
        cell = tf.contrib.rnn.LSTMCell(num_units)
        cells.append(cell)
    stack = tf.contrib.rnn.MultiRNNCell(cells)

    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden, NUM_CLASSES], stddev=0.1))

    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[NUM_CLASSES]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, NUM_CLASSES])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    # TODO: isn't seq_len is the length of each label sequence in the batch?
    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                           0.9).minimize(cost)

    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    #decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, beam_width=10)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(
        tf.edit_distance(tf.cast(decoded[0], tf.int32), targets)
    )

print("*** STARTING TRAINING SESSION ***")

with tf.Session(graph=graph) as session:
    # Initializate the weights and biases
    tf.global_variables_initializer().run()

    #session.run([targets], {targets: train_targets})
    #exit()

    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()

        # Currently we work with batches of one.
        for batch in range(num_batches_per_epoch):

            feed = {inputs: train_inputs,
                    targets: train_targets,
                    seq_len: tt_seq_len}

            batch_cost, _ = session.run([cost, optimizer], feed)
            train_cost += batch_cost*batch_size
            train_ler += session.run(ler, feed_dict=feed)*batch_size

        train_cost /= num_examples
        train_ler /= num_examples

        #val_feed = {inputs: val_inputs,
        #            targets: val_targets,
        #            seq_len: val_seq_len}

        #val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)
        val_cost, val_ler = train_cost, train_ler

        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                         val_cost, val_ler, time.time() - start))

        # Decoding
        #d = session.run(decoded[0], feed_dict=val_feed)

        #str_decoded = ''.join([MORSE_CHR[x] for x in np.asarray(d[1])]).replace('\0', '')

        #print('Original: "%s"' % ''.join(raw_targets))
        #print('Decoded:  "%s"\n' % str_decoded)
