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
#CHUNK_SIZE = 2048  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Audio sampling rate (Hz)
SIGMA = 200 # Sigma for gaussian kernel
THRESHOLD = 0 # Threshold for peak detection
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
    freq = 2**((note - 69) / 12) * 440
    #print('target:', freq)
    return freq


# Game setup
# Load resources

SCALE = 0.1
WINDOW_HEIGHT=1000
WINDOW_WIDTH=1000
app_window = pg.window.Window(WINDOW_HEIGHT,WINDOW_WIDTH)

player_image = pg.resource.image('player.png')

background_image = pg.resource.image('background.jpg')
background = pg.sprite.Sprite(background_image, x=0, y=0)
#background.height = WINDOW_HEIGHT
#background.width = WINDOW_WIDTH

class Player:

    def __init__(self, x, y):
        self.score = 0
        self.freq = 0
        self.rect = pg.shapes.Rectangle(height=50, width=50, x=x, y=y, color=(255, 255, 255))

    def move(self, y):
        y = abs(y)
        if y < 0:
            self.rect.y = 0
        elif y > WINDOW_HEIGHT:
            self.rect.y = WINDOW_HEIGHT
        else:
            self.rect.y = y

    def compare_freq(self, freq):
        # Checks if a soundblock x coordinate is withing the player
        # Also checks if the players y coordinate is within the soundblock with a small margin
        if any((block.rect.x < self.rect.x + self.rect.width and block.rect.x + block.rect.width >= self.rect.x + self.rect.width) and (self.rect.y <= block.rect.y + block.rect.height*1.3 and self.rect.y >= block.rect.y - block.rect.height*0.3)  for block in SoundBlock.soundblocks):
            self.score += 1

    # Function to get input frequency
    # The Returned frequency still has some issues when no clear input is given
    # The frequency is not stable and jumps around a lot
    # But when you f.e. talk into the microphone, the frequency is stable and (mostly) correct
    def get_input_freq(self,data):
        
        #print('too quiet')
        # Ignore data if it is too quiet to reduce background noise
        if np.max(data) > THRESHOLD:
            # Apply a Gaussian kernel to the data 
            kernel = signal.windows.gaussian(CHUNK_SIZE//10, SIGMA) # create a kernel
            kernel /= np.sum(kernel) # normalize the kernel so it does not affect the signal's amplitude
            
            #data = np.convolve(data, kernel, 'same')

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
            print(self.freq)

class SoundBlock:
        soundblocks = []


        def __init__(self, width, height, x, y):
            self.rect = pg.shapes.Rectangle(height=height, width=width, x=x, y=y, color=(100, 100, 0, 100))
    
        def move(x):
            for block in SoundBlock.soundblocks:
                block.rect.x -= x

        def draw_all():
            for block in SoundBlock.soundblocks:
                block.rect.draw()

       
# Global variables
player = Player(100, 100)
music = MidiFile("freude.mid").play()
current_msg = next(music)
prev_note_time = time.time()
game_over = False
game_started = False

@app_window.event
def on_key_press(symbol, modifiers):
    global game_over
    global current_msg
    global prev_note_time
    global music
    global game_started
    if symbol == pg.window.key.SPACE:
        if not game_started:
            game_started = True
        if game_over:
            game_over = False
            player.score = 0
            music = MidiFile("freude.mid")#.play()
            current_msg = next(music)
            prev_note_time = time.time()
            SoundBlock.soundblocks = []

@app_window.event
def on_draw():
    global current_msg
    global player
    global prev_note_time
    global game_over
    global music
    global game_started
    app_window.clear()
    background.draw()
    if not game_started:
        pg.text.Label("Press SPACE to start", x=WINDOW_WIDTH/2, y=WINDOW_HEIGHT/2).draw()
        return
    if game_over:
        pg.text.Label("DONE. Your Score: " + str(player.score), x=WINDOW_WIDTH/2, y=WINDOW_HEIGHT/2).draw()
        pg.text.Label("Press SPACE to restart", x=WINDOW_WIDTH/2, y=WINDOW_HEIGHT/2 - 50).draw()
        return
    else:
        player.rect.draw()
        SoundBlock.draw_all()
        print('----------------------')
        data = stream.read(CHUNK_SIZE)
        data = np.frombuffer(data, dtype=np.int16)

        # Handle player movement and SoundBlock movement
        player.get_input_freq(data)
        player.move(player.freq)
        # Only try to spawn soundblock if the current note is far enough away from last note
        if current_msg.time < time.time() - prev_note_time:
            prev_note_time = time.time()
            if current_msg.type == 'note_on':
                #print(current_msg)
                freq = midi_to_freq(current_msg.note)
                SoundBlock.soundblocks.append(SoundBlock(current_msg.time*200, 50, WINDOW_WIDTH, freq))
                player.compare_freq(freq)
            try:
                current_msg = next(music)
            except:
                if(player.rect.x > SoundBlock.soundblocks[-1].rect.x + SoundBlock.soundblocks[-1].rect.width):
                    game_over = True
            # AS: why 100 as magic number? shouldn't this be correlated to delta time since last update?
            SoundBlock.move(current_msg.time * 100)

        pg.text.Label("Score: " + str(player.score), x=WINDOW_WIDTH/2, y=WINDOW_HEIGHT - 30).draw()

pg.app.run()
