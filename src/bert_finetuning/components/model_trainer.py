from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from datasets import load_from_disk
from src.bert_finetuning.entity import ModelTrainerConfig, DataTransformationConfig
import os

from src.bert_finetuning.utils.common import read_yaml
from src.bert_finetuning.constants import *

import wandb

class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig,
                 data_transformation_config: DataTransformationConfig):

        self.config = config
        self.data_transformation_config = data_transformation_config
        self.training_args = read_yaml(PARAMS_FILE_PATH)

    def train(self):

        wandb.init(project = "bert-finetuning-conll2003", name="bert-NER-analysis")

        tokenizer = AutoTokenizer.from_pretrained(
            self.data_transformation_config.tokenizer_name
        )

        dataset_pt = load_from_disk(self.config.data_path)

        labels_feature = dataset_pt["train"].features["labels"]
        
        if hasattr(labels_feature, "feature") and hasattr(labels_feature.feature, "names"):
            num_labels = len(labels_feature.feature.names)
        else:
            all_labels = [label for seq in dataset_pt["train"]["labels"] for label in seq]
            num_labels = len(set(all_labels))
        
        model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_ckpt,
            num_labels=num_labels
        )

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            report_to="wandb",
            **self.training_args["TrainingArguments"]
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            label_pad_token_id=-100
        )

        trainer = Trainer(
            model=model,
            args=trainer_args,
            processing_class=tokenizer,
            data_collator=data_collator,
            train_dataset=dataset_pt["train"],
            eval_dataset=dataset_pt["validation"]
        )

        trainer.train()

        trainer.save_model(os.path.join(self.config.root_dir, "model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))