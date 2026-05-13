"""
Voice worker — IPC subprocess for the Rust orchestrator.

Loads Vosk (speech-to-text) and RoBERTa (intent classifier) once at startup,
then streams microphone audio continuously. When a complete utterance is
detected it classifies the intent and writes a single JSON line to stdout.

Protocol (stdout, newline-delimited JSON):
  {"type": "ready"}
  {"type": "utterance", "text": "go get the butter", "intent": "get butter"}
"""

import json
import os
import sys

import pyaudio
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from vosk import KaldiRecognizer
from vosk import Model as VoskModel

VOSK_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "../speech-to-text/vosk/vosk-model-small-en-us-0.15"
)
ROBERTA_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "../sentence-classifier/finetuned_roberta"
)

# Must match the label order used during RoBERTa fine-tuning.
INTENTS = [
    "get butter",
    "perform generic task",
    "answer question",
    "existential crisis",
    "seeking companionship",
]

SAMPLE_RATE = 16000
CHUNK_SIZE = 4096


def emit(msg: dict) -> None:
    print(json.dumps(msg), flush=True)


def load_classifier(model_path: str):
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer


def classify(model, tokenizer, text: str) -> str:
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
    return INTENTS[logits.argmax(-1).item()]


def main() -> None:
    print("[Voice Worker] Loading Vosk...", file=sys.stderr)
    vosk_model = VoskModel(VOSK_MODEL_PATH)
    recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)

    print("[Voice Worker] Loading RoBERTa...", file=sys.stderr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, tokenizer = load_classifier(ROBERTA_MODEL_PATH)
    classifier.to(device)

    mic = pyaudio.PyAudio()
    stream = mic.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=8192,
    )
    stream.start_stream()

    emit({"type": "ready"})

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    intent = classify(classifier, tokenizer, text)
                    emit({"type": "utterance", "text": text, "intent": intent})
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        mic.terminate()


if __name__ == "__main__":
    main()
