from vosk import Model, KaldiRecognizer
import pyaudio
import json
import os

model_path = os.path.join(os.path.dirname(__file__), "vosk", "vosk-model-small-en-us-0.15")
model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

mic = pyaudio.PyAudio()

stream = mic.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=2,  # MacBook Pro Microphone
                frames_per_buffer=8192)
stream.start_stream()
print("Listening... (speak now, Ctrl+C to stop)")

while True:
    data = stream.read(4096, exception_on_overflow=False)
    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        text = result.get("text", "").strip()
        if text:
            print(f"Heard: {text}")
    else:
        partial = json.loads(recognizer.PartialResult())
        partial_text = partial.get("partial", "").strip()
        if partial_text:
            print(f"  ... {partial_text}", end="\r")