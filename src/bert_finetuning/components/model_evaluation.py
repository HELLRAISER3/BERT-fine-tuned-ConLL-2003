import os
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from src.bert_finetuning.entity import ModelEvaluationConfig
from evaluate import load


def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [item["labels"] for item in batch], batch_first=True, padding_value=-100
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def calculate_metric_on_test_ds(self, dataset, metric, model, label_names, batch_size=16):
        dataloader = DataLoader(
            dataset.with_format("torch"),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        model.eval()
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            predictions = outputs.logits.argmax(dim=-1).cpu()

            true_predictions = [
                [label_names[p] for p, l in zip(pred, label) if l != -100]
                for pred, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_names[l] for l in label if l != -100]
                for label in labels
            ]

            metric.add_batch(predictions=true_predictions, references=true_labels)

        return metric.compute()

    def evaluate(self):
        dataset_pt = load_from_disk(self.config.data_path)

        # fallback if ClassLabel didn't persist
        CONLL2003_LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

        labels_feature = dataset_pt["test"].features["labels"]
        if hasattr(labels_feature, "feature") and hasattr(labels_feature.feature, "names"):
            label_names = labels_feature.feature.names
        else:
            label_names = CONLL2003_LABELS

        model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_path,
            num_labels=len(label_names)
        ).to(self.device)

        score = self.calculate_metric_on_test_ds(
            dataset_pt["test"], load("seqeval"), model, label_names, batch_size=16
        )

        metrics_dict = {
            "overall_precision": score["overall_precision"],
            "overall_recall": score["overall_recall"],
            "overall_f1": score["overall_f1"],
            "overall_accuracy": score["overall_accuracy"],
        }

        df = pd.DataFrame(metrics_dict, index=["bert-NER"])
        df.to_csv(self.config.metric_file_name, index=False)
        print(f"\nEvaluation Results:\n{df.to_string()}")