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


def check_for_whistle(freq_history):
    # turn freq_history into numpy array to process data
    freq_history = np.array(freq_history)
    # remove all frequencies below the threshold to get rid of outliers

    try:
        freq_history = freq_history[freq_history > WHISTLE_THRESHOLD]
    except TypeError:
        print('error')
        return

    # if there are not enough samples, do nothing
    if len(freq_history) > 5:

        # split the freq_history into two halves
        freq_history1 = freq_history[:len(freq_history)//2]
        freq_history2 = freq_history[len(freq_history)//2:]
        # if first half mean is lower than second half mean the whistle is going up
        # then press the right arrow key
        # else press the left arrow key

        # AS: why doesn't it stop?
        if np.mean(freq_history1) < np.mean(freq_history2):
            print('right')
            pynput.keyboard.Controller().press(pynput.keyboard.Key.right)
        else:
            print('left')
            pynput.keyboard.Controller().press(pynput.keyboard.Key.left)

# timer setup
whistle_started = False
whistle_started_time = time.time()

while True:

    # Read audio data from stream
    data = stream.read(CHUNK_SIZE)

    # Convert audio data to numpy array
    data = np.frombuffer(data, dtype=np.int16)

    # Get frequency
    freq = get_frequency(data)

    # if the detected frequency is above the threshold and a whistle detection has not started yet, start the whistle timer
    if freq is not None:
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


