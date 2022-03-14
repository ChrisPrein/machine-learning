from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic
from datetime import timedelta, datetime
from enum import Enum
from ...modeling.abstractions.model import Model
from .evaluation_context import EvaluationContext, TModel

TEvaluationContext = TypeVar('TEvaluationContext', EvaluationContext)

class EvaluationMetric(Generic[TEvaluationContext], ABC):
    
    @abstractmethod
    def calculate_score(self, context: TEvaluationContext) -> float:
        pass