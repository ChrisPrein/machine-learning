from abc import ABC, abstractmethod
from typing import *

from ...modeling.abstractions.model import *
from .evaluation_metric import EvaluationMetric
from .default_evaluation_plugin import *

DATASET = Iterable[Iterable[Tuple[TInput, TTarget]]]
EVALUATION_DATASET = Union[Tuple[str, DATASET], DATASET, Dict[str, DATASET], Iterable[DATASET]]
EVALUATION_METRICS = Dict[str, EvaluationMetric[TInput, TTarget]]

class EvaluationService(Generic[TInput, TTarget, TModel], ABC):
    @overload
    async def evaluate(self, model: TModel, evaluation_dataset: DATASET, evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> Dict[str, Score]: ...
    @overload
    async def evaluate(self, model: TModel, evaluation_dataset: Tuple[str, DATASET], evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> Dict[str, Score]: ...
    @overload
    async def evaluate(self, model: TModel, evaluation_dataset: Dict[str, DATASET], evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> Dict[str, Score]: ...
    @overload
    async def evaluate(self, model: TModel, evaluation_dataset: Iterable[DATASET], evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> Dict[str, Score]: ...
    @abstractmethod
    async def evaluate(self, model: TModel, evaluation_dataset: EVALUATION_DATASET, evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> Dict[str, Score]: ...