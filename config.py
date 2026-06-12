# Sound device used to play samples
DEVICE = 0

# Used audio sampling rate
FRAMERATE = 8000

# The size of the FFT (no FFT used yet)
FFT_SIZE = 128 # 62.5Hz wide bins

# Size of buffer processed by the neural network in a single step
CHUNK = 256

# The size of a sample in chunks
SEQ_LENGTH = (FRAMERATE * 12) // CHUNK * CHUNK
TIMESTEPS = SEQ_LENGTH // CHUNK

# The character set in canonical order
MORSE_CHR = [
    ' ',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    '0','1','2','3','4','5','6','7','8','9',
    '.',
    ',',
    '?',
    '/',
    '=',
    '+',
    '\0'
]

NUM_CLASSES = len(MORSE_CHR)

# Maps a character to it's serial number
MORSE_ORD = {}
for i in range(len(MORSE_CHR)):
    MORSE_ORD[MORSE_CHR[i]] = i

# Characters and morse code representations
CHARS = {
    'A': '.-',
    'B': '-...',
    'C': '-.-.',
    'D': '-..',
    'E': '.',
    'F': '..-.',
    'G': '--.',
    'H': '....',
    'I': '..',
    'J': '.---',
    'K': '-.-',
    'L': '.-..',
    'M': '--',
    'N': '-.',
    'O': '---',
    'P': '.--.',
    'Q': '--.-',
    'R': '.-.',
    'S': '...',
    'T': '-',
    'U': '..-',
    'V': '...-',
    'W': '.--',
    'X': '-..-',
    'Y': '-.--',
    'Z': '--..',
    '0': '-----',
    '1': '.----',
    '2': '..---',
    '3': '...--',
    '4': '....-',
    '5': '.....',
    '6': '-....',
    '7': '--...',
    '8': '---..',
    '9': '----.',
    '.': '.-.-.-',
    ',': '--..--',
    '?': '..--..',
    '/': '-..-.',
    '=': '-...-',
    '+': '.-.-.',
    ' ': None
}

# Training data generations

SAMPLE_GENERATOR_WORKERS=10

# Training configuration

BATCH_SIZE = 100
NUM_BATCHES_PER_EPOCH = 60
MAX_EPOCHS = 10000
CHECKPOINT_DIR = './model_train'
LOG_DIR = './logs'
L2_LAMBDA = 0.005
GRADIENT_CLIP_NORM = 1.0
CHECKPOINTS_TO_KEEP = 1000
