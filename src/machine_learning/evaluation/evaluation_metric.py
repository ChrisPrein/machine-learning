from abc import ABC, abstractmethod
from typing import Generic, Iterable
from .evaluation_context import Prediction
from ..modeling.model import TInput, TTarget, TOutput

__all__ = ['EvaluationMetric']

class EvaluationMetric(Generic[TInput, TTarget, TOutput], ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def update(self, batch: Iterable[Prediction[TInput, TTarget, TOutput]]): ...

    @property
    @abstractmethod
    def score(self) -> float: ...

    def __call__(self, batch: Iterable[Prediction[TInput, TTarget, TOutput]]):
        return self.update(batch)