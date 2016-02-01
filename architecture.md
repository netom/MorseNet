# CW decoder neural network architecture

- LSTM: 10-100-1000 input?
- LRU deep layers: 0-3
- Softmax output layer
  - The output needs to be normalized
    - A linear layer estimates probabilities (z = W^T * h + b)
   - Then softmax = exp(z_i)/(summa(exp(z_j)))

- Training according to the cross-entropy (negative log-likelyhood)

