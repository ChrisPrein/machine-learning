from abc import abstractmethod
from typing import *
from .evaluation_context import Prediction
from ..modeling.model import TInput, TTarget, TOutput
from .evaluation_metric import EvaluationMetric

__all__ = ['MultiMetric']

class MultiMetric(Generic[TInput, TTarget, TOutput], EvaluationMetric[TInput, TTarget, TOutput]):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def scores(self) -> Dict[str, float]:
        pass

    def __call__(self, batch: List[Prediction[TInput, TTarget, TOutput]]):
        return self.update(batch)