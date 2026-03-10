from src.bert_finetuning.config.configuration import ConfigurationManager
from src.bert_finetuning.components.model_evaluation import ModelEvaluation
from src.bert_finetuning.logging import logger


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.evaluate()