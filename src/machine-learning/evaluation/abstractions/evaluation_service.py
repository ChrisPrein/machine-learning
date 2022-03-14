from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Dict
from datetime import timedelta, datetime
from enum import Enum
from ...modeling.abstractions.model import Model
from .evaluation_metric import TEvaluationContext, EvaluationMetric
from .evaluation_context import EvaluationContext, TModel
from torch.utils.data import DataLoader

class EvaluationService(Generic[TEvaluationContext], ABC):
    
    @abstractmethod
    async def evaluate(self, model: TModel, evaluation_data_loader: DataLoader, evaluation_metrics: Dict[str, EvaluationMetric[TEvaluationContext]]) -> Dict[str, float]:
        pass