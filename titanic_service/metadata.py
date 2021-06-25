from titanic_service.serializer import ModelSerializer


class ModelMetadata:

    def __init__(self):
        self._models_metadata = None
        self.serializer = ModelSerializer()

    @property
    def models_metadata(self) -> dict:
        if self._models_metadata is None:
            self._models_metadata = {
                model_id: self.serializer.get_metadata(model_id)
                for model_id in self.serializer.trained_models_ids
            }
        return self._models_metadata
