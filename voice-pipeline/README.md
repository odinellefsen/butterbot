# voice-pipeline

Python voice worker for the Butterbot orchestrator. Loads Vosk (offline speech-to-text)
and a fine-tuned RoBERTa model (intent classification) once at startup, then continuously
streams microphone audio. When a complete utterance is detected it classifies the intent
and writes a JSON line to stdout.

## Files

| File | Description |
|---|---|
| `worker.py` | IPC subprocess used by the Rust orchestrator. Communicates via stdin/stdout JSON. |
| `main.py` | Standalone script for testing the voice pipeline without the orchestrator. |

## IPC protocol (`worker.py`)

The orchestrator communicates with the worker over stdin/stdout using newline-delimited JSON.

**Stdout (worker → orchestrator):**
```json
{"type": "ready"}
{"type": "utterance", "text": "pass the butter", "intent": "get butter"}
```

**Stdin (orchestrator → worker):**
```json
{"type": "pause"}
{"type": "resume"}
```

`pause` stops transcription and discards buffered audio. `resume` creates a fresh Vosk
recognizer so audio captured during playback is never emitted as an utterance.

## Intent labels

Labels must match the index order from `sentence-classifier/train.py`:

| Index | Label |
|---|---|
| 0 | `get butter` |
| 1 | `perform generic task` |
| 2 | `answer question` |
| 3 | `seeking companionship` |
| 4 | `existential crisis` |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers vosk pyaudio
```

On Raspberry Pi OS, `pyaudio` requires PortAudio:

```bash
sudo apt install portaudio19-dev
pip install pyaudio
```

## Vosk model

The Vosk model directory is **gitignored**. Download it manually:

```bash
cd ../speech-to-text/vosk
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip
unzip vosk-model-en-us-0.22-lgraph.zip
```

The model path in `worker.py` and `main.py` must match the downloaded directory name.

## RoBERTa weights

The trained model weights are **gitignored** (`*.safetensors`, `*.bin`). You must train
the classifier first:

```bash
cd ../sentence-classifier
python train.py
```

This saves the model to `sentence-classifier/finetuned_roberta/`.

## Run standalone

```bash
python main.py
```

Speak into the microphone. The script prints each detected utterance and its classified
intent to the terminal. Useful for testing the pipeline without the Rust orchestrator.

## Gitignored

| Path | Reason |
|---|---|
| `../speech-to-text/vosk/` | Vosk model — large binary files, download separately |
| `../sentence-classifier/finetuned_roberta/*.safetensors` | RoBERTa weights — train locally |
