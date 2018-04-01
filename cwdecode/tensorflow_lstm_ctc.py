import time

import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np

from config import *

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

# Hyper-parameters
num_epochs = 200
num_hidden = 50
num_units = 50
num_layers = 1
initial_learning_rate = 1e-2
momentum = 0.9

num_examples = 101
batch_size = 1
num_batches_per_epoch = int(num_examples/batch_size)

# Load the data

target_filename_tpl = 'training_set/%04d/%03d.txt'
audio_filename_tpl  = 'training_set/%04d/%03d.wav'

train_inputs  = []
train_targets = []
raw_targets   = []

# Files must be of the same length in one batch
for i in range(num_batches_per_epoch):
    audio_filename = audio_filename_tpl % (i // 35, i % 35)

    fs, audio = wav.read(audio_filename)

    time_steps = len(audio)//CHUNK
    truncated_autio_length = time_steps * CHUNK

    # Input shape is [num_batches, time_steps, CHUNK (features)]
    inputs = np.reshape(audio[:truncated_autio_length],  (1, time_steps, CHUNK))
    inputs = (inputs - np.mean(inputs)) / np.std(inputs)

    train_inputs.append(inputs)

# Convert inputs to numpy array and normalize them
# This piece of code tries to assemble a batch, but for now it assembles
# several batches of length of one.
train_seq_len = [train_inputs[0].shape[1]]

# Read targets
for i in range(num_batches_per_epoch):
    target_filename = target_filename_tpl % (i // 35, i % 35)

    with open(target_filename, 'r') as f:
        targets = list(map(lambda x: x[0], f.readlines()))

    raw_targets.append(''.join(targets))

    # Transform char into index
    targets = np.asarray([MORSE_CHR.index(x) for x in targets])

    # Creating sparse representation to feed the placeholder
    train_targets.append(sparse_tuple_from([targets]))

# Build a sparse matrix for training required by the ctc loss function
#train_targets = sparse_tuple_from(train_targets)
#train_targets = np.asarray(train_targets)

# Make our validation set to be the 0th one
val_inputs, val_targets, val_seq_len = train_inputs[0], train_targets[0], train_seq_len


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
    seq_len = tf.placeholder(tf.int32, [None])

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell 
    cells = []
    for _ in range(num_layers):
        cell = tf.contrib.rnn.LSTMCell(num_units)  # Or LSTMCell(num_units)
        cells.append(cell)
    stack = tf.contrib.rnn.MultiRNNCell(cells)

    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

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

    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                           0.9).minimize(cost)

    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))

with tf.Session(graph=graph) as session:
    # Initializate the weights and biases
    tf.global_variables_initializer().run()

    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()

        for batch in range(1,num_batches_per_epoch):

            # Currently we work with batches of one.
            feed = {inputs: train_inputs[batch],
                    targets: train_targets[batch],
                    seq_len: train_seq_len}

            batch_cost, _ = session.run([cost, optimizer], feed)
            train_cost += batch_cost*batch_size
            train_ler += session.run(ler, feed_dict=feed)*batch_size

        train_cost /= num_examples
        train_ler /= num_examples

        val_feed = {inputs: val_inputs,
                    targets: val_targets,
                    seq_len: val_seq_len}

        val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                         val_cost, val_ler, time.time() - start))

        # Decoding
        d = session.run(decoded[0], feed_dict=val_feed)

        str_decoded = ''.join([MORSE_CHR[x] for x in np.asarray(d[1])]).replace('\0', '')

        print('Original: %s' % raw_targets[0])
        print('Decoded:  %s\n' % str_decoded)
