from src.bert_finetuning.config.configuration import ConfigurationManager
from src.bert_finetuning.components.model_trainer import ModelTrainer
from src.bert_finetuning.logging import logger


class ModelTrainerTrainPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        data_transformation_config = config.get_data_transformation_config()
        model_trainer = ModelTrainer(config=model_trainer_config, 
                                     data_transformation_config=data_transformation_config)
        model_trainer.train()