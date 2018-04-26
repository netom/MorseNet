# TODO

* Try using some kind of feature extraction. Real-valued DFT looks
  promising, since it can radically decrease the amount of training
  data (the signal is very narrow band, only a few DFT buckets
  contain most of the energy).
* Make prediction made happen: read raw audio, try to decode singals.
* Train a network to identify multiple signals in audio data and
  mark them with their approximate frequencies.
* ???
* Profit.
