from titanic_service.serializer import ModelSerializer


class ModelRepository:

    @staticmethod
    def get_model(model_id: int):
        return ModelSerializer().get_model(model_id)
