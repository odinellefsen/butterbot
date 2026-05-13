from transformers import RobertaTokenizer, RobertaForSequenceClassification
from vosk import Model as VoskModel, KaldiRecognizer
import torch
import pyaudio
import json
import os
import subprocess
import sys
import random

# Paths
VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../speech-to-text/vosk/vosk-model-en-us-0.22-lgraph")
ROBERTA_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../sentence-classifier/finetuned_roberta")
VOICE_DIR = os.path.join(os.path.dirname(__file__), "../butterbot_voice")
READY_SOUNDS = [
    os.path.join(VOICE_DIR, "what_is_my_purpose.mp3"),
    os.path.join(VOICE_DIR, "what_is_my_purpose_2.mp3"),
]
ACTION_SOUNDS = {
    "perform generic task":    [os.path.join(VOICE_DIR, "i_am_not_programmed_for_that.mp3")],
    "answer question":         [os.path.join(VOICE_DIR, "i_am_not_programmed_for_questions.mp3")],
    "seeking companionship":   [os.path.join(VOICE_DIR, "i_am_not_programmed_for_friendship.mp3")],
    "existential crisis":      [os.path.join(VOICE_DIR, "oh_my_god.mp3"),
                                os.path.join(VOICE_DIR, "you_are_fucking_kidding_me.mp3")],
}


def play_sound(path):
    if sys.platform == "darwin":
        subprocess.Popen(["afplay", path])
    else:
        subprocess.Popen(["mpg123", "-q", path])

# Order must match label indices used during training (see sentence-classifier/train.py).
ACTIONS = ["get butter", "perform generic task", "answer question", "seeking companionship", "existential crisis"]


def load_classifier(model_path):
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer


def predict_action(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
    return ACTIONS[logits.argmax(-1).item()]


def main():
    print("Loading speech recognition model...")
    vosk_model = VoskModel(VOSK_MODEL_PATH)
    recognizer = KaldiRecognizer(vosk_model, 16000)

    print("Loading action classifier...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, tokenizer = load_classifier(ROBERTA_MODEL_PATH)
    classifier.to(device)

    mic = pyaudio.PyAudio()
    stream = mic.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        input_device_index=2,  # MacBook Pro Microphone
        frames_per_buffer=8192,
    )
    stream.start_stream()
    print("Ready. Speak now.\n")
    play_sound(random.choice(READY_SOUNDS))

    try:
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    action = predict_action(classifier, tokenizer, text)
                    print(f'Heard:  "{text}"')
                    print(f"Action: {action}\n")
                    if action == "get butter":
                        play_sound(random.choice(READY_SOUNDS))
                    else:
                        play_sound(random.choice(ACTION_SOUNDS[action]))
            else:
                partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                if partial:
                    print(f"  ... {partial}", end="\r")
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        stream.stop_stream()
        stream.close()
        mic.terminate()


if __name__ == "__main__":
    main()
