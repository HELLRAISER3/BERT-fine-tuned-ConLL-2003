from src.bert_finetuning.entity import DataIngestionConfig
from datasets import load_dataset
from src.bert_finetuning.logging import logger
from src.bert_finetuning.utils.common import get_size
import os
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_dataset(self):
        dataset = load_dataset(self.config.dataset_checkpoint)
        if not os.path.exists(self.config.local_data_file):
            dataset.save_to_disk(self.config.local_data_file)
            logger.info(f"{self.config.dataset_checkpoint} is downloaded!")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
        
        return dataset