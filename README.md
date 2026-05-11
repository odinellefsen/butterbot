# butterbot

A collection of components for building a butter robot.

## Components

| Directory | Description |
|---|---|
| [`sentence-classifier/`](./sentence-classifier) | Fine-tuned RoBERTa model that maps input phrases to robot actions |

---

## sentence-classifier

Classifies input phrases into one of four actions:

- **I get butter** — neutral butter request
- **I get butter, you're mean** — butter request with a rude tone
- **I get butter, I'm happy** — butter request with a kind tone
- **perform generic task** — everything else

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install transformers torch
```

### Train

Fine-tunes `roberta-base` and saves the model to `sentence-classifier/finetuned_roberta`:

```bash
cd sentence-classifier
python train.py
```

Training runs for 150 epochs on CPU. Model weights are **not** committed to the repo (too large for GitHub).

### Run inference

```bash
python sentence-classifier/phrase_action_match.py
```

Edit `input_phrase` in `phrase_action_match.py` to test different phrases.

> **Note:** You must train the model first (or obtain the `finetuned_roberta/` weights separately) before running inference.
