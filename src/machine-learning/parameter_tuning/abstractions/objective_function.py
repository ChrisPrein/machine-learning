from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from enum import Enum
from ...modeling.abstractions.model import Model
from ...evaluation.abstractions.evaluation_metric import EvaluationMetric, TModel

class OptimizationType(Enum):
    MIN = 1,
    MAX = 2


class ObjectiveFunction(EvaluationMetric[TModel], ABC):
    @property
    @abstractmethod
    def optimization_type(self) -> OptimizationType:
        pass
