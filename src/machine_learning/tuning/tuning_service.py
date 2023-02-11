from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Callable, Dict, Optional, TypeVar, Generic, Tuple, Union, overload, Iterable

from ..modeling.model import TInput, TTarget, Model
from ..training import TrainingService

__all__ = ['TuningService', 'TModel', 'Dataset', 'TrainingDataset']

TModel = TypeVar('TModel', bound=Model)

Dataset = Iterable[Iterable[Tuple[TInput, TTarget]]]
TrainingDataset = Union[Dataset[TInput, TTarget], Tuple[str, Dataset[TInput, TTarget]]]

class TuningService(Generic[TInput, TTarget, TModel], ABC):
    @abstractmethod
    async def tune(self, training_function: Callable[[Dict[str, Any]], None], params: Dict[str, Any], logger: Optional[Logger] = None) -> None: ...