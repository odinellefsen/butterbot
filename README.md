# Butterbot

An autonomous butter-fetching robot inspired by the butter-passing robot from Rick and Morty.
Built on a Raspberry Pi 5 with a Python/Rust hybrid architecture: Python handles ML inference,
Rust orchestrates the system as a state machine.

## Directories

| Directory | Description |
|---|---|
| [`orchestrator/`](./orchestrator) | Rust state machine — the robot's brain. Spawns and coordinates all Python workers via stdin/stdout JSON IPC, controls the listening lifecycle, drives audio playback, and will handle GPIO/motors on the Pi. |
| [`voice-pipeline/`](./voice-pipeline) | Python voice worker. Loads Vosk (speech-to-text) and a fine-tuned RoBERTa model (intent classification). Runs as a subprocess of the orchestrator, streaming microphone audio and emitting classified utterances as JSON. |
| [`sentence-classifier/`](./sentence-classifier) | RoBERTa fine-tuning code. Trains the intent classifier on five categories: `get butter`, `perform generic task`, `answer question`, `seeking companionship`, and `existential crisis`. The trained model is saved to `finetuned_roberta/`. |
| [`butter_detection_yolov8/`](./butter_detection_yolov8) | Python vision worker. Loads a custom-trained YOLOv8 ONNX model and runs real-time butter detection from the Pi camera. Emits bounding box coordinates to the orchestrator. Runs as a stub on non-Pi platforms. |
| [`speech-to-text/`](./speech-to-text) | Vosk offline speech recognition model files. The `vosk/` subdirectory contains the downloaded model used by the voice pipeline worker. |
| [`butterbot_voice/`](./butterbot_voice) | Robot voice clips (MP3). Played by the orchestrator in response to classified intents — includes startup lines, rejection responses, and existential crisis audio. |

## Architecture

```
┌─────────────────────────────────────────────┐
│              Rust Orchestrator               │
│                (State Machine)               │
│                                              │
│  Booting → Ready → Activating → Listening   │
│       ↓ Classifying ↓                       │
│  Scanning → Approaching → Delivering        │
│  Responding / ExistentialCrisis → Ready     │
└────────────┬─────────────────┬──────────────┘
             │  JSON over IPC  │
     ┌───────┴──────┐  ┌───────┴──────┐
     │ Voice Worker │  │ Vision Worker│
     │ Vosk+RoBERTa │  │ YOLOv8 ONNX │
     └──────────────┘  └──────────────┘
```

## Running

```bash
cd orchestrator
cargo run
```

Both Python workers are spawned automatically. Press **Enter** to activate the robot (simulates the physical button). On a Raspberry Pi the button press will be wired to a GPIO pin instead.
