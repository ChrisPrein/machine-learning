from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Dict
from datetime import timedelta, datetime
from enum import Enum
from ...abstractions.model import Model
from .evaluation_metric import TModel, EvaluationMetric

class EvaluationService(Generic[TModel], ABC):
    
    @abstractmethod
    def evaluate(self, model: TModel, evaluation_metrics: Dict[str, EvaluationMetric[TModel]]) -> Dict[str, float]:
        pass