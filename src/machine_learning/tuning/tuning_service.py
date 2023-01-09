from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Callable, Dict, Optional, TypeVar, Generic, Tuple, Union, overload, Iterable

from ..evaluation.multi_evaluation_context import Score
from ..modeling.model import TInput, TTarget, Model
from ..training import TrainingService

__all__ = ['TuningService', 'TModel', 'Dataset', 'TrainingDataset']

TModel = TypeVar('TModel', bound=Model)

Dataset = Iterable[Iterable[Tuple[TInput, TTarget]]]
TrainingDataset = Union[Dataset[TInput, TTarget], Tuple[str, Dataset[TInput, TTarget]]]

class TuningService(Generic[TInput, TTarget, TModel], ABC):
    
    @overload
    async def tune(self, model_factory: Callable[[], TModel], training_service_factory: Callable[[], TrainingService[TInput, TTarget, TModel]], train_dataset: Dataset[TInput, TTarget], params: Dict[str, Any], logger: Optional[Logger] = None) -> Dict[str, Any]: ...
    @overload
    async def tune(self, model_factory: Callable[[], TModel], training_service_factory: Callable[[], TrainingService[TInput, TTarget, TModel]], train_dataset: Tuple[str, Dataset[TInput, TTarget]], params: Dict[str, Any], logger: Optional[Logger] = None) -> Dict[str, Any]: ...
    @abstractmethod
    async def tune(self, model_factory: Callable[[], TModel], training_service_factory: Callable[[], TrainingService[TInput, TTarget, TModel]], train_dataset: TrainingDataset[TInput, TTarget], params: Dict[str, Any], logger: Optional[Logger] = None) -> Dict[str, Any]: ...