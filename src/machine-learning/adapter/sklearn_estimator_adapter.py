from ..abstractions.model import Model
from sklearn.base import BaseEstimator
from typing import Dict, Any, TypeVar, Generic
import copy

class SkleanEstimatorAdapter(BaseEstimator):
    def __init__(self, prototype_model: Model=None, **sk_params):
        self.sk_params = sk_params
        self.prototype_model: Model = prototype_model
        self.model: Model = self.prototype_model.__class__(self.sk_params)

    def fit(self, X, y, **kwargs):
        self.model.train(X, y)
        
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, **params):
        res = copy.deepcopy(self.sk_params)
        res.update({'prototype_model': self.prototype_model})
        return res

    def set_params(self, **params):
        self.sk_params.update(params)
        return self