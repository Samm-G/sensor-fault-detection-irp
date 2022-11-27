from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
import os,sys
from sensor.logger import logging
from sensor.pipeline import training_pipeline
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils.main_utils import read_yaml_file, set_env_variables

from sensor.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from sensor.constant.training_pipeline import SAVED_MODEL_DIR, TARGET_COLUMN
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.utils.main_utils import load_object

from fastapi import FastAPI
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from uvicorn import run as app_run

from sensor.constant.application import APP_HOST, APP_PORT
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd

def main():
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

app = FastAPI()
origins=["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == '__main__':
    set_env_variables()
    app_run(app, host=APP_HOST, port=APP_PORT)

# API Calls..
@app.get('/', tags=['authentication'])
async def index():
    return RedirectResponse(url='/docs')

@app.get('/train')
async def trainRouteClient():
    try:
        set_env_variables()
        if TrainPipeline.is_pipeline_running:
            return Response('Training Pipeline already running!')
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
        return Response('Training Successfull')
    except Exception as e:
        print(e)
        logging.exception(e)

@app.get('/predict')
async def predictRouteClient():
    try:
        # TODO: Get data from user csv file
        # Convert csv to Dataframe
        user_df = pd.Dataframe()

        # Use Train df
        # Upload DF
        # Calculate Data Drift.
            # If significant, trigger an email to data scientist team that Data Drift is detected.

        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.does_model_exist:
            return Response('Model not available')
        
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(best_model_path)
        y_pred = model.predict(user_df)
        pred_column = y_pred.copy()
        pred_column.replace(TargetValueMapping().reverse_mapping(), inplace=True)

        # Decide how to return file to user.


        return Response('Predicition Successfull')
    except Exception as e:
        print(e)
        logging.exception(e)