from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Dict, Tuple
from ...modeling.abstractions.model import Model, TInput, TTarget
from .evaluation_metric import TEvaluationContext, EvaluationMetric
from .evaluation_context import EvaluationContext, TModel
from dataset_handling.dataloader import DataLoader

class EvaluationService(Generic[TInput, TTarget, TModel, TEvaluationContext], ABC):
    
    @abstractmethod
    async def evaluate(self, model: TModel, evaluation_data_loader: DataLoader[Tuple[TInput, TTarget]], evaluation_metrics: Dict[str, EvaluationMetric[TEvaluationContext]]) -> Dict[str, float]:
        pass