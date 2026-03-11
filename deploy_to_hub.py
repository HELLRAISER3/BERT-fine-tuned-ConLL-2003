import argparse
import csv
from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer
from huggingface_hub import HfApi

# conll-2003 label mapping
ID2LABEL = {
    0: "O",
    1: "B-PER", 2: "I-PER",
    3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC",
    7: "B-MISC", 8: "I-MISC",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

MODEL_PATH = "artifacts/model_trainer/model"
TOKENIZER_PATH = "artifacts/model_trainer/tokenizer"
METRICS_PATH = "artifacts/model_evaluation/metrics.csv"


def load_metrics():
    with open(METRICS_PATH) as f:
        reader = csv.DictReader(f)
        return next(reader)


def build_model_card(repo_id: str, metrics: dict) -> str:
    precision = float(metrics["overall_precision"])
    recall = float(metrics["overall_recall"])
    f1 = float(metrics["overall_f1"])
    accuracy = float(metrics["overall_accuracy"])

    return f"""---
language: en
license: apache-2.0
base_model: bert-base-cased
tags:
  - bert
  - token-classification
  - ner
  - conll2003
datasets:
  - conll2003
metrics:
  - seqeval
pipeline_tag: token-classification
---

# BERT fine-tuned on CoNLL-2003 (NER)

`bert-base-cased` fine-tuned for Named Entity Recognition on [CoNLL-2003](https://huggingface.co/datasets/conll2003).

Recognizes 4 entity types: **PER**, **ORG**, **LOC**, **MISC**.

## Evaluation results

| Metric    | Score  |
|-----------|--------|
| Precision | {precision:.4f} |
| Recall    | {recall:.4f} |
| F1        | {f1:.4f} |
| Accuracy  | {accuracy:.4f} |

Evaluated with [seqeval](https://github.com/chakki-works/seqeval) on the CoNLL-2003 test split.

## Usage

```python
from transformers import pipeline

ner = pipeline("ner", model="{repo_id}", aggregation_strategy="simple")
ner("Elon Musk founded SpaceX in California.")
```

## Training details

- **Base model:** `bert-base-cased`
- **Dataset:** CoNLL-2003
- **Epochs:** 1
- **Effective batch size:** 16 (gradient accumulation)
- **Optimizer:** AdamW, weight decay 0.01
- **Warmup steps:** 500

## Label scheme

```
O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
```
"""


def deploy(repo_id: str):
    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATH,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    metrics = load_metrics()
    model_card = build_model_card(repo_id, metrics)

    print(f"Pushing to {repo_id}...")
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Done ://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF repo id")
    args = parser.parse_args()
    deploy(args.repo)
