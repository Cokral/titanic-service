from titanic_service.serializer import ModelSerializer


class ModelExperiment:

    def __init__(self):
        self.serializer = ModelSerializer()

    def log_model(self, model):
        self.serializer.save_model(model)

    def log_metadata(self, parameters: dict, metrics: dict) -> dict:
        metadata = {**parameters, **metrics}
        self.serializer.save_metadata(metadata)
        return metadata


