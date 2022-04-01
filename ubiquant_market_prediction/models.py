from abc import abstractclassmethod
import lightgbm as lgb
import numpy as np


def get_model(model_type, model_args):
    models = {
        "random": RandomModel,
        "lightgbm": LightGBMModel,
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


class LightGBMModel:
    def __init__(self, categorical_feature="auto", **lightgbm_params):
        self.categorical_feature = categorical_feature
        self.lightgbm_params = lightgbm_params

    def fit(self, x, y):
        train_set = lgb.Dataset(x, y, categorical_feature=self.categorical_feature)

        # train
        self.engine = lgb.train(
            self.lightgbm_params,
            train_set,
            None,
        )

    def predict(self, x):
        return self.engine.predict(x)
