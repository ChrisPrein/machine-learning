from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic
from dataclasses import dataclass
from ...modeling.abstractions.model import Model, TInput, TTarget
from ..contexts.evaluation_context import *

class EvaluationMetric(Generic[TInput, TTarget, TModel], ABC):
    
    @abstractmethod
    def calculate_score(self, context: EvaluationContext[TInput, TTarget, TModel]) -> float:
        pass

    def __call__(self, context: EvaluationContext[TInput, TTarget, TModel]) -> float:
        return self.calculate_score(context)