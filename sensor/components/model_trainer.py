from sensor.utils.main_utils import load_numpy_array_data
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from sensor.entity.config_entity import ModelTrainerConfig
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel
from sensor.utils.main_utils import save_object, load_object
import os, sys

from xgboost import XGBClassifier

class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SensorException(e,sys)

    def perform_hyper_parameter_tuning(self):
        # TODO: Can do Tuning Later..
        pass

    def train_model(self, x_train, y_train):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x_train,y_train)
            return xgb_clf
        except Exception as e:
            raise SensorException(e,sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Loading Trian and Test Array..
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )

            model = self.train_model(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            classification_train_metric = get_classification_score(y_true= y_train,y_pred= y_train_pred)
            classification_test_metric = get_classification_score(y_true= y_test,y_pred= y_test_pred)

            # Check Overfitting/Underfitting..
            diff_f1_score = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
            
            if diff_f1_score <= self.model_trainer_config.over_under_fitting_threshold:
                raise Exception(f"Overfitting/Underfitting Model!, Model is not good try to do more experimentation. F1 Score Difference {diff_f1_score} is less than cutoff {self.model_trainer_config.over_under_fitting_threshold}")
            
            if classification_train_metric.f1_score < self.model_trainer_config.expected_accuracy:
                raise Exception(f"Train Model is not good to provide expected accuracy. F1 Score: {classification_train_metric.f1_score}, Expected Score: {self.model_trainer_config.expected_accuracy}")

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            sensor_model = SensorModel(preprocessor=preprocessor, model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=sensor_model)

            # Model Trainer Artifact:
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric)

            logging.info(f'Model Trainer Artifact: {model_trainer_artifact}')
            return model_trainer_artifact


        except Exception as e:
            raise SensorException(e,sys)
        pass