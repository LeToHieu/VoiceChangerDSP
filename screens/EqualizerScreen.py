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

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = 1024 * 8

audio = pyaudio.PyAudio()

stream_input = None
stream_output = None
is_streaming = False
pitch_factor = 0
is_recording = False
recorded_frames = []

BANDS = [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
equalizer_factors = {}
for freq in BANDS:
    if freq >= 1000:
        key = f"{int(freq/1000)}k"
    else:
        key = str(freq)
    equalizer_factors[key] = {"gain": 1.0, "freq": freq}

effects_factors = {
    "Echo": {"gain": 0.0, "buffer": np.zeros(CHUNK)},
    "Reverb": {"gain": 0.0, "buffer": np.zeros(CHUNK * 4)},
    "Delay": {"gain": 0.0, "buffer": np.zeros(CHUNK)},
    "Distortion": {"gain": 0.0},
    "Volume": {"gain": 1.0}
}

echo_buffer = collections.deque(maxlen=int(RATE * 0.5))
delay_buffer = collections.deque(maxlen=int(RATE * 0.5))
reverb_buffer = collections.deque(maxlen=int(RATE * 1.0))

def start_stream():
    global stream_input, stream_output, is_streaming, pitch_factor, equalizer_factors, status_indicator

    if not is_streaming:
        stream_input = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        stream_output = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=CHUNK
        )

        is_streaming = True
        status_indicator.configure(text_color="green")
        threading.Thread(target=stream_audio, daemon=True).start()

def stop_stream():
    global stream_input, stream_output, is_streaming, status_indicator

    if is_streaming:
        is_streaming = False
        stream_input.stop_stream()
        stream_input.close()

        stream_output.stop_stream()
        stream_output.close()

        status_indicator.configure(text_color="red")
        print("Streaming stopped")

def stream_audio():
    global pitch_factor
    while is_streaming:
        try:
            audio_data = stream_input.read(CHUNK, exception_on_overflow=False)

            audio_data = np.frombuffer(audio_data, dtype=np.float32)
            
            audio_data = process_with_equalizer(audio_data)
            
            audio_data = change_pitch(audio_data, RATE, pitch_factor)
            
            if is_recording:
                recorded_frames.append(audio_data.copy())
            
            stream_output.write(audio_data.astype(np.float32).tobytes())
            
        except Exception as e:
            print(f"Stream error: {e}")
            continue

def change_pitch(audio_data, rate, factor):
    y_shifted = librosa.effects.pitch_shift(audio_data, sr=rate, n_steps=factor)
    return y_shifted

def process_with_equalizer(data):
    try:
        fft_data = fft(data)
        frequencies = np.fft.fftfreq(len(fft_data), 1.0/RATE)
        
        for i, center_freq in enumerate(BANDS):
            if i == 0:
                mask = np.abs(frequencies) <= center_freq
            elif i == len(BANDS) - 1:
                mask = np.abs(frequencies) > BANDS[i-1]
            else:
                mask = (np.abs(frequencies) > BANDS[i-1]) & (np.abs(frequencies) <= center_freq)
            
            band_key = str(center_freq) if center_freq < 1000 else f"{int(center_freq/1000)}k"
            
            fft_data[mask] *= equalizer_factors[band_key]["gain"]
        
        processed_data = np.real(np.fft.ifft(fft_data))
        
        if effects_factors["Echo"]["gain"] > 0:
            processed_data = apply_echo(processed_data, effects_factors["Echo"]["gain"])
        
        if effects_factors["Reverb"]["gain"] > 0:
            processed_data = apply_reverb(processed_data, effects_factors["Reverb"]["gain"])
            
        if effects_factors["Delay"]["gain"] > 0:
            processed_data = apply_delay(processed_data, effects_factors["Delay"]["gain"])
            
        if effects_factors["Distortion"]["gain"] > 0:
            processed_data = apply_distortion(processed_data, effects_factors["Distortion"]["gain"])
        
        processed_data = apply_volume(processed_data, effects_factors["Volume"]["gain"])
        
        if np.max(np.abs(processed_data)) > 1.0:
            processed_data = processed_data / np.max(np.abs(processed_data))
        
        return processed_data
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return data

def update_equalizer(band, value):
    global equalizer_factors
    gain = float(value) / 25.0
    equalizer_factors[band]["gain"] = gain

def apply_echo(data, gain):
    buffer = effects_factors["Echo"]["buffer"]
    output = data + buffer * gain
    
    effects_factors["Echo"]["buffer"] = data
    
    return output

def apply_reverb(data, gain):
    buffer = effects_factors["Reverb"]["buffer"]
    decay = 0.5
    
    output = data.copy()
    for i in range(4):
        delay_idx = i * CHUNK
        output += buffer[delay_idx:delay_idx + CHUNK] * (decay ** (i+1)) * gain
    
    effects_factors["Reverb"]["buffer"] = np.roll(buffer, -CHUNK)
    effects_factors["Reverb"]["buffer"][-CHUNK:] = data
    
    return output

def apply_delay(data, gain):
    buffer = effects_factors["Delay"]["buffer"]
    output = data + buffer * gain
    
    effects_factors["Delay"]["buffer"] = data + buffer * 0.3
    
    return output

def apply_distortion(data, gain):
    drive = 1.0 + gain * 10.0
    
    data = data * drive
    data = np.clip(data, -1.0, 1.0)
    data = np.sin(data * np.pi/2)
    
    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))
    
    return data * gain

def apply_volume(data, gain):
    return data * gain

def update_effect(effect, value):
    gain = float(value) / 100.0
    effects_factors[effect]["gain"] = gain

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
                wf.setsampwidth(2)
                wf.setframerate(RATE)
                for frame in recorded_frames:
                    frame_data = (frame * 32767).astype(np.int16)
                    wf.writeframes(frame_data.tobytes())
            print(f"Recording saved to: {file_path}")
        except Exception as e:
            print(f"Error saving recording: {e}")

def EqualizerScreen(parent):
    header_frame = ctk.CTkFrame(parent)
    header_frame.pack(pady=20)

    label = ctk.CTkLabel(header_frame, text="Equalizer", font=("Arial", 24))
    label.pack(side="left", padx=(0, 10))

    global status_indicator
    status_indicator = ctk.CTkLabel(
        header_frame,
        text="●",
        font=("Arial", 24),
        text_color="red",
        width=30,
        height=30
    )
    status_indicator.pack(side="left", padx=5)

    control_frame = ctk.CTkFrame(parent)
    control_frame.pack(expand=True, fill="both", pady=10, padx=10)

    control_frame.grid_columnconfigure(0, weight=2)
    control_frame.grid_columnconfigure(1, weight=1)

    control_frame.grid_rowconfigure(0, weight=1)
    control_frame.grid_rowconfigure(1, weight=2)
    control_frame.grid_rowconfigure(2, weight=1)

    pitch_frame = ctk.CTkFrame(control_frame)
    pitch_frame.grid(row=0, column=0, padx=20, pady=20)

    pitch_header = ctk.CTkFrame(pitch_frame)
    pitch_header.pack(fill="x", padx=5)

    pitch_label = ctk.CTkLabel(pitch_header, text="Voice pitch:", font=("Arial", 14))
    pitch_label.pack(side="left", pady=10)

    global record_button
    record_button = ctk.CTkButton(
        pitch_header,
        text="⬛",
        text_color="white",
        width=30,
        height=30,
        command=lambda: stop_recording() if is_recording else start_recording()
    )
    record_button.pack(side="right", pady=10, padx=5)

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

    pitch_slider = ctk.CTkSlider(pitch_frame, from_=-12, to=12, orientation="horizontal")
    pitch_slider.set(0)
    pitch_slider.pack(fill="x", padx=5, pady=(0, 10))

    def update_pitch(value):
        global pitch_factor
        pitch_factor = float(value)
        
        factor = 2 ** (value/12)
        factor_label.configure(text=f"×{factor:.1f}")
        
        if value > 4:
            icon_label.configure(text=" 🦫 ")
        elif value < -4:
            icon_label.configure(text=" 🤖 ")
        else:
            icon_label.configure(text=" 😊 ")

    pitch_slider.configure(command=update_pitch)

    equalizer_frame = ctk.CTkFrame(control_frame)
    equalizer_frame.grid(row=1, column=0, padx=10, pady=0)

    bands = ["32", "64", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]

    sliders = {}
    db_labels = {}

    for band in bands:
        band_frame = ctk.CTkFrame(equalizer_frame)
        band_frame.pack(side="left", padx=5)

        db_label = ctk.CTkLabel(
            band_frame, 
            text="0 dB",
            font=("Arial", 10),
            width=45,
            height=20,
            fg_color=("gray85", "gray25"),
            corner_radius=5
        )
        db_label.pack(pady=(5, 0))
        db_labels[band] = db_label

        slider = ctk.CTkSlider(
            band_frame, from_=0, to=100, orientation="vertical", height=200
        )
        slider.set(50)
        slider.pack(pady=(5, 5))

        band_label = ctk.CTkLabel(band_frame, text=band, font=("Arial", 10))
        band_label.pack(pady=5)

        sliders[band] = slider

        def update_equalizer_and_label(value, b=band, label=db_labels[band]):
            gain = float(value) / 25.0
            
            if gain > 0:
                db = 20 * np.log10(gain)
                db = int(round(db))
                label.configure(text=f"{db:+d} dB")
            else:
                label.configure(text="-∞ dB")
            
            equalizer_factors[b]["gain"] = gain

        slider.configure(command=lambda value, b=band, label=db_labels[band]: 
                       update_equalizer_and_label(value, b, label))

        update_equalizer_and_label(50, band, db_labels[band])

    controls_frame = ctk.CTkFrame(control_frame)
    controls_frame.grid(row=0, column=1, padx=10, pady=10)

    start_button = ctk.CTkButton(controls_frame, text="Start", command=start_stream)
    start_button.pack(pady=10)

    stop_button = ctk.CTkButton(controls_frame, text="Stop", command=stop_stream)
    stop_button.pack(pady=10)

    effect_frame = ctk.CTkFrame(control_frame)
    effect_frame.grid(row=1, column=1, padx=10, pady=0)
    effects = ["Echo", "Reverb", "Delay", "Distortion", "Volume"]
    
    effect_labels = {}
    
    LABEL_WIDTH = 40
    SLIDER_WIDTH = 150
    VALUE_WIDTH = 50
    
    for effect in effects:
        effect_frame_c = ctk.CTkFrame(effect_frame)
        effect_frame_c.pack(pady=5, fill="x")

        row_frame = ctk.CTkFrame(effect_frame_c)
        row_frame.pack(fill="x", padx=5, pady=5)

        effect_label = ctk.CTkLabel(row_frame, text=effect, font=("Arial", 10), 
                                  width=LABEL_WIDTH, anchor="w")
        effect_label.pack(side="left", padx=(5, 10))

        effect_slider = ctk.CTkSlider(row_frame, from_=0, to=100, width=SLIDER_WIDTH)
        initial_value = 50 if effect == "Volume" else 0
        effect_slider.set(initial_value)
        effect_slider.pack(side="left", padx=5)

        value_label = ctk.CTkLabel(row_frame, text=f"{initial_value}%", font=("Arial", 10), 
                                 width=VALUE_WIDTH, anchor="w")
        value_label.pack(side="left", padx=5)
        effect_labels[effect] = value_label

        def update_effect_and_label(value, e=effect, label=value_label):
            gain = float(value) / 100.0
            effects_factors[e]["gain"] = gain
            label.configure(text=f"{int(value)}%")

        effect_slider.configure(command=lambda value, e=effect, label=value_label: 
                              update_effect_and_label(value, e, label))
