# speech-to-text

Offline speech recognition powered by [Vosk](https://alphacephei.com/vosk/). The model
runs entirely on-device with no internet connection required, which is essential for the
Raspberry Pi deployment.

## Files

| File | Description |
|---|---|
| `mic_test.py` | Quick microphone test — prints partial and final transcriptions to the terminal. Useful for verifying audio input works before running the full pipeline. |
| `vosk/` | Downloaded Vosk model directory (gitignored). |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install vosk pyaudio
```

On Raspberry Pi OS, install PortAudio first:

```bash
sudo apt install portaudio19-dev
pip install pyaudio
```

## Download the model

The `vosk/` directory is **gitignored**. Create it and download the recommended model:

```bash
mkdir -p vosk
cd vosk
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip
unzip vosk-model-en-us-0.22-lgraph.zip
```

## Model options

| Model | Size | RAM | Notes |
|---|---|---|---|
| `vosk-model-small-en-us-0.15` | 40MB | ~200MB | Fast but inaccurate — not recommended |
| `vosk-model-en-us-0.22-lgraph` | 128MB | ~500MB | Recommended — good accuracy, Pi-friendly |
| `vosk-model-en-us-0.22` | 1.8GB | ~2GB | Best accuracy, needs 4GB+ Pi |

The model path in `voice-pipeline/worker.py` and `voice-pipeline/main.py` must match the
name of the downloaded directory exactly.

## Test the microphone

```bash
python mic_test.py
```

Speak into the microphone. Partial transcriptions are printed in place as you speak;
final results are printed on a new line when Vosk detects the end of an utterance.

If no audio is detected, check your input device index. List available devices:

```python
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(i, p.get_device_info_by_index(i)['name'])
```

Update `input_device_index` in `mic_test.py` and `voice-pipeline/worker.py` to match
your microphone.

## Gitignored

| Path | Reason |
|---|---|
| `vosk/` | Vosk model files — large binaries, download separately |
