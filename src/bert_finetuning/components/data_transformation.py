import os
from src.bert_finetuning.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk
from src.bert_finetuning.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def tokenize_function(self, example_batch):
        input_encodings = self.tokenizer(
            example_batch['tokens'],
            truncation=True,
            is_split_into_words=True
        )

        all_labels = []
        for i, labels in enumerate(example_batch['ner_tags']):
            word_ids = input_encodings.word_ids(batch_index=i)
            aligned_labels = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    aligned_labels.append(-100)        
                elif word_id != prev_word_id:
                    aligned_labels.append(labels[word_id])  
                else:
                    aligned_labels.append(-100)       
                prev_word_id = word_id
            all_labels.append(aligned_labels)

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': all_labels  
        }

    def convert(self):
        dataset = load_from_disk(self.config.data_path)

        dataset_pt = dataset.map(self.tokenize_function, batched=True)

        dataset_pt = dataset_pt.remove_columns(
            ['tokens', 'pos_tags', 'chunk_tags', 'id', 'ner_tags']
        )

        dataset_pt = dataset_pt.cast_column(
            "labels",
            dataset["train"].features["ner_tags"]
        )

        dataset_pt.save_to_disk(
            os.path.join(self.config.root_dir, "dataset")
        )
