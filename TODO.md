# TODO

* Generate samples on-the-fly, without saving them to disk
* Figure out how training works with generated samples (tensorflow
  docs are not very good in this regard, or in any regard for that
  matter)
* Try using some kind of feature extraction. Real-valued DFT looks
  promising, since it can radically decrease the amount of training
  data (the signal is very narrow band, only a few DFT buckets
  contain most of the energy).
* Train a network to identify multiple signals in audio data and
  mark them with their approximate frequencies.
* ???
* Profit.
