import pyaudio
import numpy as np
from matplotlib import pyplot as plt
import pyglet as pg
import mido
from mido import MidiFile
import time
from scipy import signal


# Code for setting up audio stream and getting the audio device copied from audio-sample.py

# Set up audio stream
# reduce chunk size and sampling rate for lower latency
CHUNK_SIZE = 2048  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Audio sampling rate (Hz)
SIGMA = 200 # Sigma for gaussian kernel
THRESHOLD = 100 # Threshold for peak detection
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


# Game setup
# Load resources

SCALE = 0.3
WINDOW_HEIGHT=1000
WINDOW_WIDTH=1000
app_window = pg.window.Window(WINDOW_HEIGHT,WINDOW_WIDTH)

player_image = pg.resource.image('player.png')

background_image = pg.resource.image('background.jpg')
background = pg.sprite.Sprite(background_image, x=0, y=0)
background.height = WINDOW_HEIGHT
background.width = WINDOW_WIDTH

block_image = pg.resource.image('soundblock.png')

class Player:

    def __init__(self, x, y):
        self.score = 0
        self.freq = 0
        self.sprite = pg.sprite.Sprite(player_image, x=x, y=y)
        self.sprite.scale = SCALE

    def move(self, y):
        y = abs(y)
        if y < 0:
            self.sprite.y = 0
        elif y > WINDOW_HEIGHT:
            self.sprite.y = WINDOW_HEIGHT
        else:
            self.sprite.y = y

    def compare_freq(self, freq):
        if self.sprite.y <= freq*1.2 and self.sprite.y >= freq *0.8:
            self.score += 1
            print("Freq matched")

    # Function to get input frequency
    # The Returned frequency still has some issues when no clear input is given
    # The frequency is not stable and jumps around a lot
    # But when you f.e. talk into the microphone, the frequency is stable and correct
    def get_input_freq(self,data):
        
        # Ignore data if it is too quiet to reduce background noise
        if np.max(data) > THRESHOLD:
            # Apply a Gaussian kernel to the data 
            kernel = signal.windows.gaussian(CHUNK_SIZE//10, SIGMA) # create a kernel
            kernel /= np.sum(kernel) # normalize the kernel so it does not affect the signal's amplitude
            
            data = np.convolve(data, kernel, 'same')

            # Apply a Hamming window
            window = np.hamming(CHUNK_SIZE)
            data = data * window

            # Perform rFFT (r gets only the positive frequencies)
            fft_vals = np.fft.rfft(data)

            # Get absolute value to determine magnitude
            fft_abs = np.abs(fft_vals)

            # Get frequency list
            freqs = np.fft.rfftfreq(CHUNK_SIZE, 1.0/RATE)

            # Get peak frequency
            peak_freq = freqs[np.argmax(fft_abs)]
            self.freq = peak_freq

class SoundBlock:
    
        def __init__(self, x, y):
            self.sprite = pg.sprite.Sprite(block_image, x=x, y=y)
            self.sprite.scale = SCALE
    
        def move(self, y):
            self.sprite.y = y

       
# Global variables
player = Player(100, 100)
soundblock = SoundBlock(600, 100)
music = MidiFile("freude.mid").play()
current_msg = next(music)
prev_note_time = time.time()
game_over = False

@app_window.event
def on_key_press(symbol, modifiers):
    global game_over
    global current_msg
    global prev_note_time
    global music
    if symbol == pg.window.key.SPACE:
        if game_over:
            game_over = False
            player.score = 0
            music = MidiFile("freude.mid").play()
            current_msg = next(music)
            prev_note_time = time.time()

@app_window.event
def on_draw():
    global current_msg
    global player
    global soundblock
    global prev_note_time
    global game_over
    global music
    app_window.clear()
    background.draw()
    if game_over:
        pg.text.Label("DONE. Your Score: " + str(player.score), x=WINDOW_WIDTH/2, y=WINDOW_HEIGHT/2).draw()
        pg.text.Label("Press SPACE to restart", x=WINDOW_WIDTH/2, y=WINDOW_HEIGHT/2 - 50).draw()
        return
    else:
        player.sprite.draw()
        soundblock.sprite.draw()
        data = stream.read(CHUNK_SIZE)
        data = np.frombuffer(data, dtype=np.int16)

        # Handle player movement and SoundBlock movement
        player.get_input_freq(data)
        player.move(player.freq)
        if current_msg.time < time.time() - prev_note_time:
            prev_note_time = time.time()
            if current_msg.type == 'note_on':
                freq = midi_to_freq(current_msg.note)
                soundblock.move(freq)
                player.compare_freq(freq)
            try:
                current_msg = next(music)
            except:
                game_over = True
        
        pg.text.Label("Score: " + str(player.score), x=WINDOW_WIDTH/2, y=WINDOW_HEIGHT - 30).draw()

pg.app.run()
