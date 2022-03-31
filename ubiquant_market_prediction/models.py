from abc import abstractclassmethod
import numpy as np


def get_model(model_type, model_args):
    models = {
        "random": RandomModel,
    }
    return models[model_type.lower()](**model_args)


class BaseModel:
    @abstractclassmethod
    def fit(self, x, y):
        pass

    @abstractclassmethod
    def predict(self, x):
        pass


class RandomModel(BaseModel):
    def fit(self, x, y):
        pass

    def predict(self, x):
        return np.random.normal(size=len(x))
