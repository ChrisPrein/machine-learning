from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic
from datetime import timedelta, datetime
from enum import Enum
from ...abstractions.model import Model

TModel = TypeVar('TModel', Model)


class EvaluationMetric(Generic[TModel], ABC):
    
    @abstractmethod
    def calculate_score(self, model: TModel) -> float:
        pass