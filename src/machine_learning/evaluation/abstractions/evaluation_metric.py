from abc import ABC, abstractmethod
from typing import Generic

from ...modeling.abstractions.model import TInput, TTarget
from ..contexts.evaluation_context import *

class EvaluationMetric(Generic[TInput, TTarget], ABC):

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def update(self, batch: Iterable[Prediction[TInput, TTarget]]): ...

    @property
    @abstractmethod
    def score(self) -> float: ...

    def __call__(self, batch: Iterable[Prediction[TInput, TTarget]]):
        return self.update(batch)