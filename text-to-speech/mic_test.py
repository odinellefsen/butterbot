from vosk import Model, KaldiRecognizer
import pyaudio

model = Model(r"C:/Users/odin/repos/Butterbot/llm_test/vosk/vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)

mic = pyaudio.PyAudio()

stream = mic.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=2,  # This targets the microphone at index 2
                frames_per_buffer=8192)
stream.start_stream()


while True:
    data = stream.read(4096)
    if recognizer.AcceptWaveform(data):
        text = recognizer.Result()
        print(text[14:-3])