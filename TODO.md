# TODO

* Accuracy instead of lexical error rate
* Decoding real-time audio data
* Try using some kind of feature extraction. Real-valued DFT looks
  promising, since it can radically decrease the amount of training
  data (the signal is very narrow band, only a few DFT buckets
  contain most of the energy).
* Train a network to identify multiple signals in audio data and
  mark them with their approximate frequencies.
* ???
* Profit.

* https://keras.io/examples/audio/ctc_asr/

Sometimes:

======================================================================
Epoch 32/1000
======================================================================
Epoch  31, Batch   0/60: Loss=10.0314 (CTC=7.5063, L2=2.5251), LER=0.1202
Epoch  31, Batch  10/60: Loss=10.9510 (CTC=8.4262, L2=2.5248), LER=0.1228
Epoch  31, Batch  20/60: Loss=11.5268 (CTC=9.0023, L2=2.5245), LER=0.1263
Epoch  31, Batch  30/60: Loss=10.2694 (CTC=7.7452, L2=2.5243), LER=0.1168
Epoch  31, Batch  40/60: Loss=10.1248 (CTC=7.6008, L2=2.5240), LER=0.1044
Epoch  31, Batch  50/60: Loss=11.1482 (CTC=8.6245, L2=2.5237), LER=0.1181
2026-02-02 14:33:42.894831: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence


This crap comes up from time to time:

======================================================================
Epoch 33/1000
======================================================================
Epoch  32, Batch   0/60: Loss=12.7222 (CTC=10.1987, L2=2.5236), LER=0.1279
Epoch  32, Batch  10/60: Loss=10.3898 (CTC=7.8665, L2=2.5233), LER=0.1033
Epoch  32, Batch  20/60: Loss=9.8552 (CTC=7.3323, L2=2.5230), LER=0.1049
2026-02-02 14:34:12.799225: W tensorflow/core/framework/op_kernel.cc:1842] INVALID_ARGUMENT: ValueError: Dimensions 1 and 4746 are not compatible
Traceback (most recent call last):

  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/data/ops/from_generator_op.py", line 263, in generator_py_func
    values = next(generator_state.get_iterator(iterator_id.numpy()))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/cuccok/programozas/AI/cwdecode/./tensorflow_lstm_ctc_train.py", line 240, in generator_wrapper
    sparse_label = tf.SparseTensor(
                   ^^^^^^^^^^^^^^^^

  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/framework/sparse_tensor.py", line 154, in __init__
    indices_shape.dims[1].assert_is_compatible_with(dense_shape_shape.dims[0])

  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/framework/tensor_shape.py", line 303, in assert_is_compatible_with
    raise ValueError("Dimensions %s and %s are not compatible" %

ValueError: Dimensions 1 and 4746 are not compatible 

Traceback (most recent call last):
  File "/home/cuccok/programozas/AI/cwdecode/./tensorflow_lstm_ctc_train.py", line 338, in <module>
    main()
  File "/home/cuccok/programozas/AI/cwdecode/./tensorflow_lstm_ctc_train.py", line 310, in main
    metrics = trainer.train_epoch(train_dataset, epoch)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/cuccok/programozas/AI/cwdecode/./tensorflow_lstm_ctc_train.py", line 158, in train_epoch
    for batch_idx, (audio, labels) in enumerate(dataset):
  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/data/ops/iterator_ops.py", line 826, in __next__
    return self._next_internal()
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/data/ops/iterator_ops.py", line 776, in _next_internal
    ret = gen_dataset_ops.iterator_get_next(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/ops/gen_dataset_ops.py", line 3086, in iterator_get_next
    _ops.raise_from_not_ok_status(e, name)
  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/framework/ops.py", line 6027, in raise_from_not_ok_status
    raise core._status_to_exception(e) from None  # pylint: disable=protected-access
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tensorflow.python.framework.errors_impl.InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} ValueError: Dimensions 1 and 4746 are not compatible
Traceback (most recent call last):

  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/data/ops/from_generator_op.py", line 263, in generator_py_func
    values = next(generator_state.get_iterator(iterator_id.numpy()))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/cuccok/programozas/AI/cwdecode/./tensorflow_lstm_ctc_train.py", line 240, in generator_wrapper
    sparse_label = tf.SparseTensor(
                   ^^^^^^^^^^^^^^^^

  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/framework/sparse_tensor.py", line 154, in __init__
    indices_shape.dims[1].assert_is_compatible_with(dense_shape_shape.dims[0])

  File "/home/cuccok/programozas/AI/cwdecode/venv/lib/python3.12/site-packages/tensorflow/python/framework/tensor_shape.py", line 303, in assert_is_compatible_with
    raise ValueError("Dimensions %s and %s are not compatible" %

ValueError: Dimensions 1 and 4746 are not compatible 


         [[{{node EagerPyFunc}}]] [Op:IteratorGetNext] name:

