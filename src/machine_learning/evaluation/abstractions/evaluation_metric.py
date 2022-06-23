from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Union
from dataclasses import dataclass
from ...modeling.abstractions.model import Model, TInput, TTarget
from ..contexts.evaluation_context import *

class EvaluationMetric(Generic[TInput, TTarget], ABC):
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, batch: List[Prediction[TInput, TTarget]]):
        pass

    @property
    @abstractmethod
    def score(self) -> float:
        pass

    def __call__(self, batch: List[Prediction[TInput, TTarget]]):
        return self.update(batch)