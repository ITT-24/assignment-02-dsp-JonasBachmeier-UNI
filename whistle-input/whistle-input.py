import pyaudio
import numpy as np
from matplotlib import pyplot as plt
import pyglet as pg
import mido
from mido import MidiFile
import time
from scipy import signal
import pynput

# Set up audio stream
# reduce chunk size and sampling rate for lower latency
CHUNK_SIZE = 1024  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Audio sampling rate (Hz)
THRESHOLD = 100 # Threshold for peak detection
WHISTLE_THRESHOLD = 1500
BLOCK_AMOUNT = 5

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

WINDOW_HEIGHT=1000
WINDOW_WIDTH=1000
app_window = pg.window.Window(WINDOW_HEIGHT,WINDOW_WIDTH)

freq_history = []

# Code copied from karaoke.py / Übung 2
def get_frequency(data):

    if np.max(data) > THRESHOLD:
        kernel = signal.windows.gaussian(CHUNK_SIZE//10, 200) # create a kernel
        kernel /= np.sum(kernel) # normalize the kernel so it does not affect the signal's amplitude
        
        data = np.convolve(data, kernel, 'same')

        # Apply a Hamming window
        window = np.hamming(CHUNK_SIZE)
        data = data * window

        # Perform FFT
        fft = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(fft), 1/RATE)
        fft = np.abs(fft)

        # Find the peak frequency
        peak_freq = freqs[np.argmax(fft)]

        return peak_freq

# Code copied from last assignment
class Blocks:
    blocks = []
    selected_block = 0

    def __init__(self,width, height, x, y, color=(255, 0, 0)):
        self.rect = pg.shapes.Rectangle(width=width, height=height, x=x, y=y, color=color)
    
    def draw_all():
        for block in Blocks.blocks:
            block.rect.draw()

    def change_selection(dir="up"):
        # set currently selected block to red
        Blocks.blocks[Blocks.selected_block].rect.color = (255, 0, 0)
        if dir == "up":
            Blocks.selected_block += 1
        elif dir == "down":
            Blocks.selected_block -= 1

        # if selection goes out of bounds, loop around
        if Blocks.selected_block < 0:
            Blocks.selected_block = len(Blocks.blocks) - 1
        elif Blocks.selected_block >= len(Blocks.blocks):
            Blocks.selected_block = 0
        
        # set new selected block to green
        Blocks.blocks[Blocks.selected_block].rect.color = (0, 255, 0)


def check_for_whistle(freq_history):
    # turn freq_history into numpy array to process data
    freq_history = np.array(freq_history)
    # remove all frequencies below the threshold to get rid of outliers
    freq_history = freq_history[freq_history > WHISTLE_THRESHOLD]

    # if there are not enough samples, do nothing
    if len(freq_history) > 5:

        # split the freq_history into two halves
        freq_history1 = freq_history[:len(freq_history)//2]
        freq_history2 = freq_history[len(freq_history)//2:]
        # if first half mean is lower than second half mean the whistle is going up
        if np.mean(freq_history1) < np.mean(freq_history2):
            Blocks.change_selection("up")
        else:
            Blocks.change_selection("down")

# timer setup
whistle_started = False
whistle_started_time = time.time()

# create blocks
# first block is created with different color to signal selection
Blocks.blocks.append(Blocks(50, 50, 100, WINDOW_HEIGHT//2, (0, 255, 0)))
for i in range(BLOCK_AMOUNT):
    Blocks.blocks.append(Blocks(50, 50, 200 + i * 100, WINDOW_HEIGHT//2))

@app_window.event
def on_draw():
    global whistle_started
    global whistle_started_time
    global freq_history

    # Read audio data from stream
    data = stream.read(CHUNK_SIZE)

    # Convert audio data to numpy array
    data = np.frombuffer(data, dtype=np.int16)

    # Get frequency
    freq = get_frequency(data)

    # if the detected frequency is above the threshold and a whistle detection has not started yet, start the whistle timer
    if freq > WHISTLE_THRESHOLD and not whistle_started:
        whistle_started = True
        whistle_started_time = time.time()

    # while the whistle timer is running, add the detected frequencies to the freq_history
    if whistle_started and time.time() - whistle_started_time <= 1:
        freq_history.append(freq)

    # after the whistle timer has ended, check for a whistle and reset the freq_history
    if whistle_started and time.time() - whistle_started_time > 1:
        check_for_whistle(freq_history)
        
        whistle_started = False
        freq_history = []

    Blocks.draw_all()

    
pg.app.run()


