from abc import ABC, abstractmethod
import asyncio
from logging import Logger
from optparse import Option
from typing import Optional, TypeVar, List, Generic, Dict, Tuple, Union, overload, Iterable
from uuid import UUID

from ...modeling.abstractions.model import TInput, TTarget, Model

TModel = TypeVar('TModel', bound=Model)

DATASET = Iterable[Iterable[Tuple[TInput, TTarget]]]
TRAINING_DATASET = Union[Tuple[str, Iterable[Iterable[Tuple[TInput, TTarget]]]], Iterable[Iterable[Tuple[TInput, TTarget]]], Iterable[DATASET], Dict[str, DATASET]]
VALIDATION_DATASET = Optional[TRAINING_DATASET]

class TrainingService(Generic[TInput, TTarget, TModel], ABC):
    
    @overload
    async def train(self, model: TModel, dataset: DATASET, logger: Optional[Logger] = None) -> TModel: ...
    @overload
    async def train(self, model: TModel, dataset: Tuple[str, DATASET], logger: Optional[Logger] = None) -> TModel: ...
    @overload
    async def train(self, model: TModel, dataset: Iterable[DATASET], logger: Optional[Logger] = None) -> TModel: ...
    @overload
    async def train(self, model: TModel, dataset: Dict[str, DATASET], logger: Optional[Logger] = None) -> TModel: ...
    @abstractmethod
    async def train(self, model: TModel, dataset: TRAINING_DATASET,  logger: Optional[Logger] = None) -> TModel: ...