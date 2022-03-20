from abc import ABC, abstractmethod
from enum import Enum
from ...modeling.abstractions.model import Model
from ...evaluation.abstractions.evaluation_metric import EvaluationMetric, TModel, TEvaluationContext

class OptimizationType(Enum):
    MIN = 1,
    MAX = 2

class ObjectiveFunction(EvaluationMetric[TEvaluationContext], ABC):
    @property
    @abstractmethod
    def optimization_type(self) -> OptimizationType:
        pass
