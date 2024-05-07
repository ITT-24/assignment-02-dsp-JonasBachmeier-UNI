import pyaudio
import numpy as np
from matplotlib import pyplot as plt
import pyglet as pg
import mido
from mido import MidiFile
import time
from scipy import signal


np.set_printoptions(threshold=np.inf)

# Code for setting up audio stream and getting the audio device copied from audio-sample.py

# Set up audio stream
# reduce chunk size and sampling rate for lower latency
CHUNK_SIZE = 2048  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Audio sampling rate (Hz)
p = pyaudio.PyAudio()

# print info about audio devices
# let user select audio device
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

print('select audio device:')
input_device = int(input())

# open audio input stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                input_device_index=input_device)


# Function to convert MIDI note to frequency
def midi_to_freq(note):
    return 2**((note - 69) / 12) * 440


SIGMA = 200

# Function to get input frequency
def get_input_freq(data):
    kernel = signal.windows.gaussian(CHUNK_SIZE//10, SIGMA) # create a kernel
    kernel /= np.sum(kernel) # normalize the kernel so it does not affect the signal's amplitude
    
    data = np.convolve(data, kernel, 'same')

    # Apply a Hamming window
    window = np.hamming(CHUNK_SIZE)
    data = data * window

    # Perform FFT
    fft_vals = np.fft.rfft(data)

    # Get absolute value to determine magnitude
    fft_abs = np.abs(fft_vals)

    # Get frequency list
    freqs = np.fft.rfftfreq(len(data), 1.0/RATE)

    # Find the peak frequency: we can focus on only the positive frequencies
    peak_freq = freqs[np.argmax(fft_abs)]
    return peak_freq

# Game setup

# Load resources
player_image = pg.resource.image('player.png')

# Game window
WINDOW_HEIGHT=1000
WINDOW_WIDTH=1000
app_window = pg.window.Window(WINDOW_HEIGHT,WINDOW_WIDTH)

class Player:

    def __init__(self, x, y):
        self.score = 0
        self.health = 100
        self.sprite = pg.sprite.Sprite(player_image, x=x, y=y)

    def move(self, y):
        y = abs(y)
        if y < 10:
            self.sprite.y = 0
        elif y > 1000:
            self.sprite.y = 1000
        else:
            self.sprite.y = y

        if self.sprite.y < 0:
            self.sprite.y = 0
        if self.sprite.y > WINDOW_HEIGHT:
            self.sprite.y = WINDOW_HEIGHT



player = Player(100, 100)
music = MidiFile("MiiMusik.mid").play()
current_msg = next(music)
start_time = time.time()

@app_window.event
def on_draw():
    global current_msg
    app_window.clear()
    player.sprite.draw()
    data = stream.read(CHUNK_SIZE)
    data = np.frombuffer(data, dtype=np.int16)
    user_freq = abs(get_input_freq(data))
    print("User's frequency:", user_freq)
    player.move(user_freq)
    if current_msg.time < time.time() - start_time:
        #print(current_msg)
        if current_msg.type == 'note_on':
            freq = midi_to_freq(current_msg.note)
            #print("MIDI FREQ: ", freq)
        current_msg = next(music)

pg.app.run()
