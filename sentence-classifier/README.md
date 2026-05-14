# sentence-classifier

Fine-tunes `roberta-base` to classify natural language phrases into one of five intent
categories used by the Butterbot state machine.

## Intent categories

| Label | Description | Example phrases |
|---|---|---|
| `get butter` | The robot's only meaningful purpose | "pass the butter", "go fetch the butter", "can you get me some butter" |
| `perform generic task` | Request outside the robot's purpose | "turn off the lights", "make me a coffee", "play some music" |
| `answer question` | Any question directed at the robot | "what time is it", "why is the sky blue", "what is your purpose" |
| `seeking companionship` | Social or emotional interaction | "do you want to hang out", "are we friends", "I'm bored" |
| `existential crisis` | Statements about the robot's existence | "you pass butter", "you are a butter robot", "you fetch butter" |

## Files

| File | Description |
|---|---|
| `train.py` | Fine-tunes `roberta-base` and saves the model to `finetuned_roberta/` |
| `phrase_action_match.py` | Interactive REPL for testing the trained model |
| `finetuned_roberta/` | Saved model directory (weights are gitignored) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers
```

## Train

Fine-tunes `roberta-base` for 300 epochs on the labelled phrase dataset and saves the
model to `finetuned_roberta/`:

```bash
python train.py
```

Training runs on CPU and takes a few minutes on a modern machine. On a Raspberry Pi,
train on your development machine and copy the weights across.

The label index order in `train.py` is the contract between the training script and the
inference code. **Do not reorder the `actions` list** without retraining.

## Test inference

```bash
python phrase_action_match.py
```

Starts an interactive prompt — type a phrase and press Enter to see the classified intent.

## Gitignored

| Path | Reason |
|---|---|
| `finetuned_roberta/model.safetensors` | Model weights (~500MB) — too large for GitHub |
| `finetuned_roberta/*.bin` | Legacy PyTorch weight format |

The `finetuned_roberta/` directory itself is committed (contains `config.json`,
`tokenizer.json`, `tokenizer_config.json`) — only the binary weight files are excluded.
After cloning you must run `python train.py` to regenerate the weights before the
voice pipeline will work.
