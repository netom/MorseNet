# Sound device used to play samples
DEVICE = 0

# Used audio sampling rate
FRAMERATE = 882 # Roughly 50Hz wide bins

# Size of buffer processed by the neural network in a single step
CHUNK = 128

# The number of examples to generate
SETSIZE = 100

# The size of a sample in chunks (10 seconds)
SAMPLE_CHUNKS = FRAMERATE * 10 // CHUNK
# Learn on approx. 5 sec backwards
LSTM_TRUNCATED_GRADIENT = SAMPLE_CHUNKS / 3

# Size of a minibatch
MINIBATCH_SIZE = 20

# The directory in wich the examples are saved
TRAINING_SET_DIR = 'training_set'

# The character set in canonical order
MORSE_CHR = ['\0','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9', ' ']

# Maps a character to it's serial number
MORSE_ORD = {}
for i in xrange(len(MORSE_CHR)):
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
    ' ': None
}
