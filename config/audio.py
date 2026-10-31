# Used audio sampling rate
FRAMERATE = 8000

# The size of the FFT (no FFT used yet)
FFT_SIZE = 128 # 62.5Hz wide bins

# Size of buffer processed by the neural network in a single step
CHUNK = 256

# The size of a sample in chunks
SEQ_LENGTH_SECONDS = 12
SEQ_LENGTH_FRAMES = (FRAMERATE * SEQ_LENGTH_SECONDS) // CHUNK * CHUNK
TIMESTEPS = SEQ_LENGTH_FRAMES // CHUNK
