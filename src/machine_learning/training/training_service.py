from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional, TypeVar, Generic, Tuple, Union, overload, Iterable
from ..modeling.model import TInput, TTarget, Model

__all__ = ['TModel', 'Dataset', 'TrainingDataset', 'TrainingService']

TModel = TypeVar('TModel', bound=Model)

Dataset = Iterable[Iterable[Tuple[TInput, TTarget]]]
TrainingDataset = Union[Dataset[TInput, TTarget], Tuple[str, Dataset[TInput, TTarget]]]

class TrainingService(Generic[TInput, TTarget, TModel], ABC):
    
    @overload
    async def train(self, model: TModel, dataset: Dataset[TInput, TTarget], logger: Optional[Logger] = None) -> TModel: ...
    @overload
    async def train(self, model: TModel, dataset: Tuple[str, Dataset[TInput, TTarget]], logger: Optional[Logger] = None) -> TModel: ...
    @abstractmethod
    async def train(self, model: TModel, dataset: TrainingDataset[TInput, TTarget],  logger: Optional[Logger] = None) -> TModel: ...