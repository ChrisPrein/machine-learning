from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional, TypeVar, Generic, Tuple, Union, overload, Iterable
from ...modeling.abstractions.model import TInput, TTarget, Model

TModel = TypeVar('TModel', bound=Model)

DATASET = Iterable[Iterable[Tuple[TInput, TTarget]]]
TRAINING_DATASET = Union[DATASET, Tuple[str, DATASET]]

class TrainingService(Generic[TInput, TTarget, TModel], ABC):
    
    @overload
    async def train(self, model: TModel, dataset: DATASET, logger: Optional[Logger] = None) -> TModel: ...
    @overload
    async def train(self, model: TModel, dataset: Tuple[str, DATASET], logger: Optional[Logger] = None) -> TModel: ...
    @abstractmethod
    async def train(self, model: TModel, dataset: TRAINING_DATASET,  logger: Optional[Logger] = None) -> TModel: ...