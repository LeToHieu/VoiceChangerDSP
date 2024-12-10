import customtkinter as ctk
import pyaudio
import threading
import numpy as np
import librosa
from scipy.fftpack import fft
from scipy import signal
import queue
import collections
from tkinter import filedialog
import wave
import os

# Audio Parameters
FORMAT = pyaudio.paFloat32  # Changed to Float32 for better processing
CHANNELS = 1  # Mono audio
RATE = 44100  # Sample rate (44.1 kHz)
CHUNK = 1024 * 8  # Changed to match the equalizer implementation

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Global variables for stream control
stream_input = None
stream_output = None
is_streaming = False
pitch_factor = 0  # Initialize pitch factor
is_recording = False
recorded_frames = []

# Update equalizer structure with frequencies
BANDS = [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
equalizer_factors = {}
for freq in BANDS:
    if freq >= 1000:
        key = f"{int(freq/1000)}k"
    else:
        key = str(freq)
    equalizer_factors[key] = {"gain": 1.0, "freq": freq}

# Add these global variables after the equalizer_factors
effects_factors = {
    "Echo": {"gain": 0.0, "buffer": np.zeros(CHUNK)},
    "Reverb": {"gain": 0.0, "buffer": np.zeros(CHUNK * 4)},
    "Delay": {"gain": 0.0, "buffer": np.zeros(CHUNK)},
    "Distortion": {"gain": 0.0},
    "Volume": {"gain": 1.0}
}

# Create delay buffers for echo and delay effects
echo_buffer = collections.deque(maxlen=int(RATE * 0.5))  # 500ms max delay
delay_buffer = collections.deque(maxlen=int(RATE * 0.5))  # 500ms max delay
reverb_buffer = collections.deque(maxlen=int(RATE * 1.0))  # 1s reverb tail

# Function to start streaming audio
def start_stream():
    global stream_input, stream_output, is_streaming, pitch_factor, equalizer_factors, status_indicator

    if not is_streaming:
        # Open input stream (microphone)
        stream_input = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        # Open output stream (speaker)
        stream_output = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=CHUNK
        )

        is_streaming = True
        # Update indicator to green
        status_indicator.configure(text_color="green")
        # Start a separate thread to handle the audio streaming
        threading.Thread(target=stream_audio, daemon=True).start()

# Function to stop streaming audio
def stop_stream():
    global stream_input, stream_output, is_streaming, status_indicator

    if is_streaming:
        is_streaming = False
        # Stop input and output streams
        stream_input.stop_stream()
        stream_input.close()

        stream_output.stop_stream()
        stream_output.close()

        # Update indicator back to red
        status_indicator.configure(text_color="red")
        print("Streaming stopped")

# Function to continuously stream audio from mic to speaker
def stream_audio():
    global pitch_factor
    while is_streaming:
        try:
            # Read data from the microphone
            audio_data = stream_input.read(CHUNK, exception_on_overflow=False)

            # Convert audio data to numpy array for processing
            audio_data = np.frombuffer(audio_data, dtype=np.float32)
            
            # Apply equalizer
            audio_data = process_with_equalizer(audio_data)
            
            # Apply pitch shift
            audio_data = change_pitch(audio_data, RATE, pitch_factor)
            
            # Record if recording is active
            if is_recording:
                recorded_frames.append(audio_data.copy())
            
            # Write to speakers
            stream_output.write(audio_data.astype(np.float32).tobytes())
            
        except Exception as e:
            print(f"Stream error: {e}")
            continue

# Function to change pitch of audio data
def change_pitch(audio_data, rate, factor):
    # Audio is already normalized from float32 format
    y_shifted = librosa.effects.pitch_shift(audio_data, sr=rate, n_steps=factor)
    return y_shifted

# Add the process_audio function from message.txt
def process_with_equalizer(data):
    try:
        # Apply FFT
        fft_data = fft(data)
        frequencies = np.fft.fftfreq(len(fft_data), 1.0/RATE)
        
        # Apply equalizer gains
        for i, center_freq in enumerate(BANDS):
            # Find frequency range for this band
            if i == 0:
                mask = np.abs(frequencies) <= center_freq
            elif i == len(BANDS) - 1:
                mask = np.abs(frequencies) > BANDS[i-1]
            else:
                mask = (np.abs(frequencies) > BANDS[i-1]) & (np.abs(frequencies) <= center_freq)
            
            # Get the corresponding band key
            band_key = str(center_freq) if center_freq < 1000 else f"{int(center_freq/1000)}k"
            
            # Apply gain
            fft_data[mask] *= equalizer_factors[band_key]["gain"]
        
        # Convert back to time domain
        processed_data = np.real(np.fft.ifft(fft_data))
        
        # Apply effects in sequence
        if effects_factors["Echo"]["gain"] > 0:
            processed_data = apply_echo(processed_data, effects_factors["Echo"]["gain"])
        
        if effects_factors["Reverb"]["gain"] > 0:
            processed_data = apply_reverb(processed_data, effects_factors["Reverb"]["gain"])
            
        if effects_factors["Delay"]["gain"] > 0:
            processed_data = apply_delay(processed_data, effects_factors["Delay"]["gain"])
            
        if effects_factors["Distortion"]["gain"] > 0:
            processed_data = apply_distortion(processed_data, effects_factors["Distortion"]["gain"])
        
        # Always apply volume
        processed_data = apply_volume(processed_data, effects_factors["Volume"]["gain"])
        
        # Ensure output is normalized
        if np.max(np.abs(processed_data)) > 1.0:
            processed_data = processed_data / np.max(np.abs(processed_data))
        
        return processed_data
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return data

# Update the update_equalizer function
def update_equalizer(band, value):
    global equalizer_factors
    # Convert slider value (0-100) to gain (0.0-4.0)
    gain = float(value) / 25.0  # This makes 50 = 2.0 (neutral)
    equalizer_factors[band]["gain"] = gain

# Add these effect processing functions
def apply_echo(data, gain):
    """Simple echo effect using a single buffer"""
    buffer = effects_factors["Echo"]["buffer"]
    output = data + buffer * gain
    
    # Update buffer for next chunk
    effects_factors["Echo"]["buffer"] = data
    
    return output

def apply_reverb(data, gain):
    """Multi-tap reverb effect"""
    buffer = effects_factors["Reverb"]["buffer"]
    decay = 0.5
    
    # Create reverb by combining multiple delayed versions
    output = data.copy()
    for i in range(4):
        delay_idx = i * CHUNK
        output += buffer[delay_idx:delay_idx + CHUNK] * (decay ** (i+1)) * gain
    
    # Update buffer
    effects_factors["Reverb"]["buffer"] = np.roll(buffer, -CHUNK)
    effects_factors["Reverb"]["buffer"][-CHUNK:] = data
    
    return output

def apply_delay(data, gain):
    """Single tap delay with feedback"""
    buffer = effects_factors["Delay"]["buffer"]
    output = data + buffer * gain
    
    # Update buffer with mix of current and previous
    effects_factors["Delay"]["buffer"] = data + buffer * 0.3  # Feedback amount
    
    return output

def apply_distortion(data, gain):
    """Waveshaping distortion"""
    # Adjust the drive amount based on gain
    drive = 1.0 + gain * 10.0
    
    # Apply soft clipping
    data = data * drive
    data = np.clip(data, -1.0, 1.0)
    data = np.sin(data * np.pi/2)
    
    # Normalize output
    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))
    
    return data * gain

def apply_volume(data, gain):
    """Simple volume control"""
    return data * gain

# Add function to update effects
def update_effect(effect, value):
    # Convert slider value (0-100) to gain (0.0-1.0)
    gain = float(value) / 100.0
    effects_factors[effect]["gain"] = gain

# Add these recording functions
def start_recording():
    global is_recording, recorded_frames
    recorded_frames = []
    is_recording = True
    record_button.configure(text="⏺", text_color="red")

def stop_recording():
    global is_recording
    if is_recording:
        is_recording = False
        record_button.configure(text="⬛", text_color="white")
        save_recording()

def save_recording():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".wav",
        filetypes=[("Wave files", "*.wav")],
        title="Save Recording As"
    )
    
    if file_path:
        try:
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(RATE)
                for frame in recorded_frames:
                    frame_data = (frame * 32767).astype(np.int16)
                    wf.writeframes(frame_data.tobytes())
            print(f"Recording saved to: {file_path}")
        except Exception as e:
            print(f"Error saving recording: {e}")

# HelloScreen function with tkinter UI setup
def EqualizerScreen(parent):
    # Label at the top with status indicator
    header_frame = ctk.CTkFrame(parent)
    header_frame.pack(pady=20)

    # Label for "Equalizer"
    label = ctk.CTkLabel(header_frame, text="Equalizer", font=("Arial", 24))
    label.pack(side="left", padx=(0, 10))

    # Status indicator circle
    global status_indicator  # Make it global so we can access it from start/stop functions
    status_indicator = ctk.CTkLabel(
        header_frame,
        text="●",  # Circle symbol
        font=("Arial", 24),
        text_color="red",  # Initial color is red
        width=30,
        height=30
    )
    status_indicator.pack(side="left", padx=5)

    # Frame to hold the equalizer and controls
    control_frame = ctk.CTkFrame(parent)
    control_frame.pack(expand=True, fill="both", pady=10, padx=10)

    # Create columns in the control_frame for equalizer and settings
    control_frame.grid_columnconfigure(0, weight=2)  # Equalizer section will take 2x space
    control_frame.grid_columnconfigure(1, weight=1)  # Controls section will take 1x space

    control_frame.grid_rowconfigure(0, weight=1)  # Space for the pitch section
    control_frame.grid_rowconfigure(1, weight=2)  # Space for equalizer sliders
    control_frame.grid_rowconfigure(2, weight=1)  # Space for controls (Start, Stop, Effects)

    # Voice pitch section (row 0, column 0)
    pitch_frame = ctk.CTkFrame(control_frame)
    pitch_frame.grid(row=0, column=0, padx=20, pady=20)

    # Create a frame for pitch controls
    pitch_header = ctk.CTkFrame(pitch_frame)
    pitch_header.pack(fill="x", padx=5)

    # Label for voice pitch on the left
    pitch_label = ctk.CTkLabel(pitch_header, text="Voice pitch:", font=("Arial", 14))
    pitch_label.pack(side="left", pady=10)

    # Record button
    global record_button
    record_button = ctk.CTkButton(
        pitch_header,
        text="⬛",  # Square symbol
        text_color="white",
        width=30,
        height=30,
        command=lambda: stop_recording() if is_recording else start_recording()
    )
    record_button.pack(side="right", pady=10, padx=5)

    # Icon label for chipmunk/robot in a box
    icon_label = ctk.CTkLabel(
        pitch_header,
        text="😊",
        font=("Arial", 14),
        width=30,
        height=20,
        fg_color=("gray85", "gray25"),
        corner_radius=5
    )
    icon_label.pack(side="right", pady=10, padx=5)

    # Factor value label in a box
    factor_label = ctk.CTkLabel(
        pitch_header,
        text="×1.0",
        font=("Arial", 10),
        width=60,
        height=20,
        fg_color=("gray85", "gray25"),
        corner_radius=5
    )
    factor_label.pack(side="right", pady=10, padx=5)

    # Pitch slider with a callback function to adjust the pitch factor
    pitch_slider = ctk.CTkSlider(pitch_frame, from_=-12, to=12, orientation="horizontal")
    pitch_slider.set(0)
    pitch_slider.pack(fill="x", padx=5, pady=(0, 10))

    def update_pitch(value):
        global pitch_factor
        pitch_factor = float(value)
        
        # Calculate factor (2^(steps/12) for semitone steps)
        factor = 2 ** (value/12)
        factor_label.configure(text=f"×{factor:.1f}")
        
        # Update icon based on pitch factor
        if value > 4:
            icon_label.configure(text=" 🦫 ")  # Chipmunk icon
        elif value < -4:
            icon_label.configure(text=" 🤖 ")  # Robot icon
        else:
            icon_label.configure(text=" 😊 ")  # Human face icon for normal range

    pitch_slider.configure(command=update_pitch)

    # Equalizer section (row 1, column 0)
    equalizer_frame = ctk.CTkFrame(control_frame)
    equalizer_frame.grid(row=1, column=0, padx=10, pady=0)

    # Define equalizer bands and their frequencies
    bands = ["32", "64", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]

    # Create sliders dictionary to store references
    sliders = {}
    db_labels = {}  # Dictionary to store dB value labels

    for band in bands:
        # Create a vertical frame for each slider and its label
        band_frame = ctk.CTkFrame(equalizer_frame)
        band_frame.pack(side="left", padx=5)

        # dB value label above slider in a box
        db_label = ctk.CTkLabel(
            band_frame, 
            text="0 dB",
            font=("Arial", 10),
            width=45,  # Fixed width for the box
            height=20,  # Fixed height for the box
            fg_color=("gray85", "gray25"),  # Box background color
            corner_radius=5  # Rounded corners
        )
        db_label.pack(pady=(5, 0))
        db_labels[band] = db_label

        # Slider for the band
        slider = ctk.CTkSlider(
            band_frame, from_=0, to=100, orientation="vertical", height=200
        )
        slider.set(50)  # Set default value to the middle
        slider.pack(pady=(5, 5))

        # Frequency label for the band
        band_label = ctk.CTkLabel(band_frame, text=band, font=("Arial", 10))
        band_label.pack(pady=5)

        # Store slider reference
        sliders[band] = slider

        # Update function to show dB value
        def update_equalizer_and_label(value, b=band, label=db_labels[band]):
            # Convert slider value (0-100) to gain (0.0-4.0)
            gain = float(value) / 25.0  # This makes 50 = 2.0 (neutral)
            
            # Convert gain to dB
            if gain > 0:
                db = 20 * np.log10(gain)
                db = int(round(db))  # Round to integer
                label.configure(text=f"{db:+d} dB")  # Use integer format
            else:
                label.configure(text="-∞ dB")
            
            # Update equalizer
            equalizer_factors[b]["gain"] = gain

        # Configure slider command
        slider.configure(command=lambda value, b=band, label=db_labels[band]: 
                       update_equalizer_and_label(value, b, label))

        # Set initial dB label
        update_equalizer_and_label(50, band, db_labels[band])

    # Controls section (row 1, column 1)
    controls_frame = ctk.CTkFrame(control_frame)
    controls_frame.grid(row=0, column=1, padx=10, pady=10)

    # Start and Stop buttons
    start_button = ctk.CTkButton(controls_frame, text="Start", command=start_stream)
    start_button.pack(pady=10)

    stop_button = ctk.CTkButton(controls_frame, text="Stop", command=stop_stream)
    stop_button.pack(pady=10)

    # Effects sliders: echo, reverb, delay, distortion, volume
    effect_frame = ctk.CTkFrame(control_frame)
    effect_frame.grid(row=1, column=1, padx=10, pady=0)
    effects = ["Echo", "Reverb", "Delay", "Distortion", "Volume"]
    
    # Dictionary to store value labels
    effect_labels = {}
    
    # Set fixed widths for consistent layout
    LABEL_WIDTH = 40
    SLIDER_WIDTH = 150
    VALUE_WIDTH = 50
    
    for effect in effects:
        # Create a frame for each effect row
        effect_frame_c = ctk.CTkFrame(effect_frame)
        effect_frame_c.pack(pady=5, fill="x")

        # Create a horizontal frame to hold slider and labels
        row_frame = ctk.CTkFrame(effect_frame_c)
        row_frame.pack(fill="x", padx=5, pady=5)

        # Effect name label on the left
        effect_label = ctk.CTkLabel(row_frame, text=effect, font=("Arial", 10), 
                                  width=LABEL_WIDTH, anchor="w")
        effect_label.pack(side="left", padx=(5, 10))

        # Slider in the middle
        effect_slider = ctk.CTkSlider(row_frame, from_=0, to=100, width=SLIDER_WIDTH)
        initial_value = 50 if effect == "Volume" else 0
        effect_slider.set(initial_value)
        effect_slider.pack(side="left", padx=5)

        # Value label on the right
        value_label = ctk.CTkLabel(row_frame, text=f"{initial_value}%", font=("Arial", 10), 
                                 width=VALUE_WIDTH, anchor="w")
        value_label.pack(side="left", padx=5)
        effect_labels[effect] = value_label

        # Update function for both effect and label
        def update_effect_and_label(value, e=effect, label=value_label):
            # Update effect
            gain = float(value) / 100.0
            effects_factors[e]["gain"] = gain
            # Update label
            label.configure(text=f"{int(value)}%")

        # Configure slider command
        effect_slider.configure(command=lambda value, e=effect, label=value_label: 
                              update_effect_and_label(value, e, label))
