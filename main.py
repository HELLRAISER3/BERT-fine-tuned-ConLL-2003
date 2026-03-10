from src.bert_finetuning.pipeline.data_ingestion_01 import DataIngestionTrainingPipeline
from src.bert_finetuning.pipeline.data_validation_02 import DataValidationTrainingPipeline
from src.bert_finetuning.pipeline.data_transformation_03 import DataTransformationTrainPipeline
from src.bert_finetuning.pipeline.model_trainer_04 import ModelTrainerTrainPipeline


# Data Ingestion
# data_ingestion = DataIngestionTrainingPipeline()
# data_ingestion.main()

# Data Validation
# data_validation = DataValidationTrainingPipeline()
# data_validation.main()

# Data Transformation
# data_transformation = DataTransformationTrainPipeline()
# data_transformation.main()

# Model Training
model_trainer = ModelTrainerTrainPipeline()
model_trainer.main()
