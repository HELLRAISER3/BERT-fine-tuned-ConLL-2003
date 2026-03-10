from src.bert_finetuning.pipeline.data_ingestion_01 import DataIngestionTrainingPipeline
from src.bert_finetuning.pipeline.data_validation_02 import DataValidationTrainingPipeline
from src.bert_finetuning.pipeline.data_transformation_03 import DataTransformationTrainPipeline


# Data Ingestion
# data_ingestion = DataIngestionTrainingPipeline()
# data_ingestion.main()

# Data Validation
# data_validation = DataValidationTrainingPipeline()
# data_validation.main()

# Data Transformation
data_transformation = DataTransformationTrainPipeline()
data_transformation.main()

from datasets import load_from_disk

dataset = load_from_disk('artifacts/data_transformation/dataset')
print(dataset)