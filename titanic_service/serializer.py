import glob
import json
import os
import pickle
from titanic_service.model.model import TitanicModel



class ModelSerializer:
    BASE_PATH = 'trained_models/'

    def __init__(self):
        self._models = None
        self._models_metadata = None
        self._trained_models_ids = None
        self._next_id = None

    @property
    def trained_models_ids(self) -> list:
        if self._trained_models_ids is None:
            self._trained_models_ids = [self.extract_model_id(path) for path in glob.glob(f'{self.BASE_PATH}*')]
        return self._trained_models_ids

    @property
    def next_id(self) -> int:
        if self._next_id is None:
            self._next_id = len(self.trained_models_ids)
        return self._next_id

    @staticmethod
    def extract_model_id(path: str) -> str:
        return path.split('/')[-1]

    def make_metadata_path(self, model_id: int) -> str:
        return f'{self.BASE_PATH}{model_id}/metadata.json'

    def make_model_path(self, model_id: int) -> str:
        return f'{self.BASE_PATH}{model_id}/model.pickle'

    def make_folder_path(self, model_id: int) -> str:
        return f'{self.BASE_PATH}{model_id}'

    def get_model(self, model_id: int) -> TitanicModel:
        model_path = self.make_model_path(model_id)
        with open(model_path, 'rb') as handle:
            return pickle.load(handle)

    def get_metadata(self, model_id: str):
        metadata_path = self.make_metadata_path(model_id)
        with open(metadata_path, 'r') as handle:
            return json.load(handle)

    def save_model(self, model):
        print(glob.glob('*'))
        model_path = self.make_model_path(self.next_id)
        os.mkdir(self.make_folder_path(self.next_id))
        with open(f'{model_path}', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_metadata(self, metadata: dict):
        metadata_path = self.make_metadata_path(self.next_id)
        with open(f'{metadata_path}', 'w') as fp:
            json.dump(metadata, fp)
