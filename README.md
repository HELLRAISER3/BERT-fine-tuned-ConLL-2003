# BERT Fine-tuned on CoNLL-2003

Fine-tuning `bert-base-cased` for Named Entity Recognition (NER) on the CoNLL-2003 dataset. The dataset has 4 entity types: PER, ORG, LOC, MISC.

## Pipeline

Five sequential stages, each managed independently:

1. **Data Ingestion** — pulls CoNLL-2003 from HuggingFace Hub, saves locally as a `.hf` dataset
2. **Data Validation** — checks that train/test/validation splits all exist
3. **Data Transformation** — tokenizes with `bert-base-cased` tokenizer, handles subword alignment for NER labels
4. **Model Training** — fine-tunes BERT with HuggingFace `Trainer`, logs to W&B
5. **Model Evaluation** — runs `seqeval` metrics on the test set, saves results to CSV

## Usage

```bash
pip install -r requirements.txt
python main.py
```

Training is tracked via Weights & Biases. Set `WANDB_API_KEY` or run `wandb login` before training.

## Config

- `config/config.yaml` — paths and model checkpoints
- `params.yaml` — training hyperparameters (epochs, batch size, etc.)

Default training runs for 1 epoch with gradient accumulation (effective batch size 16) to keep memory usage low.

## Stack

- HuggingFace `transformers` + `datasets`
- `seqeval` for NER evaluation
- W&B for experiment tracking
- PyTorch backend
