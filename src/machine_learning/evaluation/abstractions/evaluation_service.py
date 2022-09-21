from abc import ABC, abstractmethod
from typing import *

from ...modeling.abstractions.model import *
from .evaluation_metric import EvaluationMetric
from .default_evaluation_plugin import *

class EvaluationService(Generic[TInput, TTarget, TModel], ABC):
    
    @abstractmethod
    async def evaluate(self, model: TModel, evaluation_dataset: Union[Tuple[str, Iterable[Tuple[TInput, TTarget]]], Iterable[Tuple[TInput, TTarget]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]], logger: Optional[Logger] = None) -> Dict[str, Score]: ...

    @abstractmethod
    async def evaluate_predictions(self, predictions: Iterable[Tuple[TInput, TTarget, TTarget]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]], logger: Optional[Logger] = None) -> Dict[str, Score]: ...

    @abstractmethod
    async def evaluate_on_multiple_datasets(self, model: TModel, evaluation_datasets: Dict[str, Iterable[Tuple[TInput, TTarget]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]]) -> Dict[str, Dict[str, Score]]: ...