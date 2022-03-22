from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, List, Generic
from ...modeling.abstractions.model import Model, TInput, TTarget

TModel = TypeVar('TModel', bound=Model)

@dataclass
class Loss:
    epoch: int
    iteration: int
    loss: float

class TrainingContext(Generic[TModel], ABC):
    
    @property
    @abstractmethod
    def model(self) -> TModel:
        pass

    @property
    @abstractmethod
    def current_epoch(self) -> int:
        pass

    @property
    @abstractmethod
    def current_iteration(self) -> int:
        pass

    @property
    @abstractmethod
    def loss(self) -> List[Loss]:
        pass