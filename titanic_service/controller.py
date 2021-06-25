import pandas as pd
from titanic_service.experiment import ModelExperiment
from titanic_service.model.model import TitanicModel
from titanic_service.serializer import ModelSerializer
from titanic_service.metadata import ModelMetadata
from titanic_service.repository import ModelRepository



class ModelController:

    def __init__(self):
        self.serializer = ModelSerializer()
        self.metadata = ModelMetadata()
        self.experiment = ModelExperiment()

    def get_models_metadata(self):
        return self.metadata.models_metadata

    def train_model(self, parameters: dict) -> dict:
        model = TitanicModel().train(parameters)
        self.experiment.log_model(model)
        metadata = self.experiment.log_metadata(parameters, model.metrics)
        return metadata

    @staticmethod
    def predict_model(features: dict, model_id: int = 0):
        model = ModelRepository.get_model(model_id)
        features = pd.DataFrame.from_dict(features, orient='index').transpose()
        prediction = model.predict(features)[0]
        return {'prediction': TitanicModel.PREDICTIONS_CLASSES[prediction]}
