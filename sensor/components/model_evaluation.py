from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from sensor.entity.config_entity import ModelEvaluationConfig
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel
from sensor.ml.model.estimator import ModelResolver
from sensor.utils.main_utils import save_object, load_object, write_yaml_file
from sensor.constant.training_pipeline import TARGET_COLUMN
import os, sys
import pandas as pd

class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig,
                data_validation_artifact: DataValidationArtifact,
                model_trainer_artifact: ModelTrainerArtifact) -> None:
        try:
            self.model_eval_config = model_eval_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise SensorException(e,sys)
        pass

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path
            
            # Get Valid Train and Test DF..
            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)
            
            # Merge DFs and then later calculate accuracy
            merged_df = pd.concat([train_df,test_df], axis=0)

            model_resolver = ModelResolver()
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path

            is_model_accepted = True
            
            if not model_resolver.does_model_exist():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    changed_accuracy=None,
                    best_model_path=None,
                    train_model_path=trained_model_file_path,
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,
                    best_model_metric_artifact=None
                    )
                logging.info(f'Model Evaluation Artifact Prepared: {model_evaluation_artifact}')
                return model_evaluation_artifact

            latest_model_path =  model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=trained_model_file_path)
            y_true = merged_df[TARGET_COLUMN]
            y_train_pred = train_model.predict(merged_df)
            y_latest_pred = latest_model.predict(merged_df)

            # Accuracy..
            trained_metric = get_classification_score(y_true, y_train_pred)
            latest_metric = get_classification_score(y_true, y_latest_pred)

            improved_accuracy = trained_metric-latest_metric
            if self.model_eval_config.change_threshold < improved_accuracy:
                is_model_accepted=True
            else:
                is_model_accepted=False

            model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    changed_accuracy=improved_accuracy,
                    best_model_path=latest_model_path,
                    train_model_path=trained_model_file_path,
                    train_model_metric_artifact=trained_metric,
                    best_model_metric_artifact=latest_metric
                    )
            model_eval_report = model_evaluation_artifact.__dict__
            write_yaml_file(self.model_eval_config.report_file_path, model_eval_report)

            logging.info(f'Model Evaluation Artifact Prepared: {model_evaluation_artifact}')

        except Exception as e:
            raise SensorException(e,sys)
        pass