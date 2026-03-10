from src.bert_finetuning.pipeline.data_ingestion_01 import DataIngestionTrainingPipeline
from src.bert_finetuning.pipeline.data_validation_02 import DataValidationTrainingPipeline
from src.bert_finetuning.pipeline.data_transformation_03 import DataTransformationTrainingPipeline
from src.bert_finetuning.pipeline.model_trainer_04 import ModelTrainerTrainingPipeline
from src.bert_finetuning.pipeline.model_evaluation_05 import ModelEvaluationTrainingPipeline


# Data Ingestion
# data_ingestion = DataIngestionTrainingPipeline()
# data_ingestion.main()

# Data Validation
# data_validation = DataValidationTrainingPipeline()
# data_validation.main()

# Data Transformation
data_transformation = DataTransformationTrainingPipeline()
data_transformation.main()

# Model Training
# model_trainer = ModelTrainerTrainingPipeline()
# model_trainer.main()

# Model Evaluation
model_evaluation = ModelEvaluationTrainingPipeline()
model_evaluation.main()