from fastapi import FastAPI
from titanic_service.model.model import TitanicPredict, TitanicParameters
from titanic_service.controller import ModelController

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/train")
def train_model(parameters: TitanicParameters):
    return ModelController().train_model(dict(parameters))

@app.get("/models")
def fetch_model_list():
    return ModelController().get_models_metadata()

@app.post("/predict")
def fetch_predictions(to_predict: TitanicPredict):
    to_predict = dict(to_predict)
    model_id = to_predict.pop('model_id')
    return ModelController().predict_model(to_predict, model_id)
