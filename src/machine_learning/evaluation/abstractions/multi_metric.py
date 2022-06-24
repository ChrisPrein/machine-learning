from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Union
from dataclasses import dataclass

from .evaluation_metric import EvaluationMetric
from ...modeling.abstractions.model import Model, TInput, TTarget
from ..contexts.evaluation_context import *

class MultiMetric(Generic[TInput, TTarget], EvaluationMetric[TInput, TTarget]):

    @property
    @abstractmethod
    def scores(self) -> Dict[str, float]:
        pass

    def __call__(self, batch: List[Prediction[TInput, TTarget]]):
        return self.update(batch)