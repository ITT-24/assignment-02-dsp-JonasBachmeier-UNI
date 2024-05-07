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

def get_frequency(data):

    kernel = signal.windows.gaussian(len(data)//10, 200) # create a kernel
    kernel /= np.sum(kernel) # normalize the kernel so it does not affect the signal's amplitude
    
    data = np.convolve(data, kernel, 'same')

    # Apply a Hamming window
    window = np.hamming(RATE)
    data = data * window

    # Perform FFT
    fft = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(fft), 1/RATE)
    fft = np.abs(fft)

    # Find the peak frequency
    peak_freq = freqs[np.argmax(fft)]

    return peak_freq

def check_for_whistle(freq_history):
    freq_history_mean = np.mean(freq_history)
    if(len(freq_history) < 100):
        return

    for i in range(len(freq_history)):
        if(freq_history[i] < freq_history_mean*0.8 or freq_history[i] > freq_history_mean*1.2):
            print("out of mean")
        else:
            print("in mean range")


# continuously capture and plot audio singal
while True:
    # Read audio data from stream
    data = stream.read(CHUNK_SIZE)

    # Convert audio data to numpy array
    data = np.frombuffer(data, dtype=np.int16)

    # Get frequency
    freq = get_frequency(data)

    if(len(freq_history) > 100):
        freq_history.pop(0)
    freq_history.append(freq)

    check_for_whistle(freq_history)
