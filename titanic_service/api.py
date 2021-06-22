from fastapi import FastAPI
from titanic_service.model.model import TitanicPredict, TitanicParameters
from titanic_service.model.utils import ModelHandler

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/train")
def train_model(parameters: TitanicParameters):
    return ModelHandler().train_and_save(dict(parameters))

@app.get("/models")
def fetch_model_list():
    return ModelHandler().models

@app.post("/predict")
def fetch_predictions(to_predict: TitanicPredict):
    to_predict = dict(to_predict)
    model_id = to_predict.pop('model_id')
    return ModelHandler().load_and_predict(to_predict, model_id)
