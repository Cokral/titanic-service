import glob
import json
import os
import pandas as pd
import pickle
from titanic_service.model.model import TitanicModel
from typing import Any


class ModelHandler:

    BASE_PATH = 'model/trained_models/'
    PREDICTIONS = {
        0: 'Survived',
        1: 'Did not survive'
    }

    def __init__(self):
        self._models = None

    @property
    def models(self) -> dict:
        if self._models is None:
            paths = glob.glob(f'{self.BASE_PATH}*')
            self._models = {
                path.split('/')[-1]: self.load_json(f'{path}/metadata.json')
                for path in paths
            }
        return self._models

    def train_and_save(self, parameters: dict) -> dict:
        model = TitanicModel().train(parameters)
        metrics = model.metrics
        folder = len(self.models)
        folder_path = f'{self.BASE_PATH}{folder}'
        os.mkdir(folder_path)
        self.save_as_pickle(model, f'{folder_path}/model')
        self.save_as_json({**parameters, **metrics}, f'{folder_path}/metadata')
        return metrics

    def load_and_predict(self, features: dict, model_id: int = 0):
        model_path = f'{self.BASE_PATH}{model_id}/model.pickle'
        model = self.load_pickle(model_path)
        features = pd.DataFrame.from_dict(features, orient='index').transpose()
        prediction = model.predict(features)[0]
        return {'prediction': self.PREDICTIONS[prediction]}

    @staticmethod
    def save_as_pickle(file: Any, filename: str):
        with open(f'{filename}.pickle', 'wb') as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle(filename: str) -> Any:
        with open(filename, 'rb') as handle:
            return pickle.load(handle)

    @staticmethod
    def save_as_json(file: Any, filename: str):
        with open(f'{filename}.json', 'w') as fp:
            json.dump(file, fp)

    @staticmethod
    def load_json(filename: str) -> Any:
        with open(filename, 'r') as handle:
            return json.load(handle)

