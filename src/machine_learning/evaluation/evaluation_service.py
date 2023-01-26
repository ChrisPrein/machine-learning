from abc import ABC, abstractmethod
from logging import Logger
from typing import *

from .multi_evaluation_context import Score
from ..modeling.model import TInput, TTarget, TOutput
from .evaluation_context import Prediction, TModel
from .evaluation_metric import EvaluationMetric
from .default_evaluation_plugin import *

__all__ = ['Dataset', 'EvaluationDataset', 'EvaluationMetrics', 'EvaluationResult', 'Predictions', 'PredictionData', 'EvaluationService']

Dataset = Iterable[Iterable[Tuple[TInput, TTarget]]]
EvaluationDataset = Union[Tuple[str, Dataset[TInput, TTarget]], Dataset[TInput, TTarget], Dict[str, Dataset[TInput, TTarget]], Iterable[Dataset[TInput, TTarget]]]
EvaluationMetrics = Dict[str, EvaluationMetric[TInput, TTarget, TOutput]]
EvaluationResult = Union[Dict[str, Score], Dict[str, Dict[str, Score]]]
Predictions = Iterable[Prediction[TInput, TTarget, TOutput]]
PredictionData = Union[Predictions[TInput, TTarget, TOutput], Tuple[str, Predictions[TInput, TTarget, TOutput]]]

class EvaluationService(Generic[TInput, TTarget, TOutput, TModel], ABC):
    @overload
    async def evaluate(self, model: TModel, evaluation_dataset: Dataset[TInput, TTarget], evaluation_metrics: EvaluationMetrics[TInput, TTarget, TOutput], logger: Optional[Logger] = None) -> Dict[str, Score]: ...
    @overload
    async def evaluate(self, model: TModel, evaluation_dataset: Tuple[str, Dataset[TInput, TTarget]], evaluation_metrics: EvaluationMetrics[TInput, TTarget, TOutput], logger: Optional[Logger] = None) -> Dict[str, Score]: ...
    @overload
    async def evaluate(self, model: TModel, evaluation_dataset: Dict[str, Dataset[TInput, TTarget]], evaluation_metrics: EvaluationMetrics[TInput, TTarget, TOutput], logger: Optional[Logger] = None) -> Dict[str, Dict[str, Score]]: ...
    @overload
    async def evaluate(self, model: TModel, evaluation_dataset: Iterable[Dataset[TInput, TTarget]], evaluation_metrics: EvaluationMetrics[TInput, TTarget, TOutput], logger: Optional[Logger] = None) -> Dict[str, Dict[str, Score]]: ...
    @abstractmethod
    async def evaluate(self, model: TModel, evaluation_dataset: EvaluationDataset[TInput, TTarget], evaluation_metrics: EvaluationMetrics[TInput, TTarget, TOutput], logger: Optional[Logger] = None) -> EvaluationResult: ...

    @overload
    async def evaluate_predictions(self, predictions: Predictions[TInput, TTarget, TOutput], evaluation_metrics: EvaluationMetrics[TInput, TTarget, TOutput], logger: Optional[Logger] = None) -> Dict[str, Score]: ...
    @overload
    async def evaluate_predictions(self, predictions: Tuple[str, Predictions[TInput, TTarget, TOutput]], evaluation_metrics: EvaluationMetrics[TInput, TTarget, TOutput], logger: Optional[Logger] = None) -> Dict[str, Score]: ...
    @abstractmethod
    async def evaluate_predictions(self, predictions: PredictionData[TInput, TTarget, TOutput], evaluation_metrics: EvaluationMetrics[TInput, TTarget, TOutput], logger: Optional[Logger] = None) -> Dict[str, Score]: ...
