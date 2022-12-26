from abc import ABC, abstractmethod
from typing import Generic, Iterable
from .evaluation_context import Prediction
from ..modeling.model import TInput, TTarget

class EvaluationMetric(Generic[TInput, TTarget], ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def update(self, batch: Iterable[Prediction[TInput, TTarget]]): ...

    @property
    @abstractmethod
    def score(self) -> float: ...

    def __call__(self, batch: Iterable[Prediction[TInput, TTarget]]):
        return self.update(batch)