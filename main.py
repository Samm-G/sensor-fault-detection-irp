from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
import os,sys
from sensor.logger import logging
from sensor.pipeline import training_pipeline
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils.main_utils import read_yaml_file, set_env_variables

from sensor.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig

if __name__ == '__main__':
    try:
        set_env_variables()
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()

        # Ingestion Only..
        #training_pipeline_config = TrainingPipelineConfig()
        #data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        #print(data_ingestion_config.__dict__)

    except Exception as e:
        print(e)
        logging.exception(e)
