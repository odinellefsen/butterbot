# butterbot

A fine-tuned RoBERTa model that classifies input phrases into one of four actions:

- **I get butter** — neutral butter request
- **I get butter, you're mean** — butter request with a rude tone
- **I get butter, I'm happy** — butter request with a kind tone
- **perform generic task** — everything else

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install transformers torch
```

## Train

Fine-tunes `roberta-base` on the labelled phrase dataset and saves the model to `./finetuned_roberta`:

```bash
python train.py
```

Training runs for 150 epochs on CPU. The model weights are saved locally and are **not** committed to the repo (too large for GitHub).

## Run inference

```bash
python phrase_action_match.py
```

Edit the `input_phrase` variable in `phrase_action_match.py` to test different phrases. The script loads the model from `./finetuned_roberta` and prints the predicted action.

> **Note:** You must train the model first (or obtain the `finetuned_roberta/` weights separately) before running inference.
