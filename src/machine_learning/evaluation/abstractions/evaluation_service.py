from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Dict, Tuple, Union
from ...modeling.abstractions.model import Model, TInput, TTarget
from .evaluation_metric import TModel, EvaluationMetric
from torch.utils.data.dataset import Dataset
from multipledispatch import dispatch

class EvaluationService(Generic[TInput, TTarget, TModel], ABC):
    
    @abstractmethod
    async def evaluate(self, model: TModel, evaluation_dataset: Union[Tuple[str, Dataset[Tuple[TInput, TTarget]]], Dataset[Tuple[TInput, TTarget]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget, TModel]]) -> Dict[str, float]:
        pass

    @abstractmethod
    async def evaluate_on_multiple_datasets(self, model: TModel, evaluation_datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget, TModel]]) -> Dict[str, Dict[str, float]]:
        pass