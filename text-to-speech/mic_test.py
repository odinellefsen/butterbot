from vosk import Model, KaldiRecognizer
import pyaudio
import os

model_path = os.path.join(os.path.dirname(__file__), "vosk", "vosk-model-small-en-us-0.15")
model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

mic = pyaudio.PyAudio()

stream = mic.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8192)
stream.start_stream()


while True:
    data = stream.read(4096, exception_on_overflow=False)
    if recognizer.AcceptWaveform(data):
        text = recognizer.Result()
        print(text[14:-3])