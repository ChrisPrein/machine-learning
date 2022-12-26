from abc import ABC, abstractmethod
from logging import Logger
from typing import Generic, Iterable, Optional, Union, overload

from ..modeling.model import TInput, TTarget
from .training_service import TModel

BATCH: Iterable[TInput]
INPUT: Union[TInput, BATCH]

class Trainer(Generic[TInput, TTarget, TModel], ABC):
    @overload
    async def train_step(self, model: TModel, input: TInput, logger: Optional[Logger] = None) -> TModel: ...
    @overload
    async def train_step(self, model: TModel, input: BATCH, logger: Optional[Logger] = None) -> TModel: ...
    @abstractmethod
    async def train_step(self, model: TModel, input: INPUT,  logger: Optional[Logger] = None) -> TModel: ...