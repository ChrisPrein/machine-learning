from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic
from .evaluation_context import EvaluationContext, TModel

TEvaluationContext = TypeVar('TEvaluationContext', bound=EvaluationContext)

class EvaluationMetric(Generic[TEvaluationContext], ABC):
    
    @abstractmethod
    def calculate_score(self, context: TEvaluationContext) -> float:
        pass