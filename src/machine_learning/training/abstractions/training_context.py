from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, List, Generic
from ...modeling.abstractions.model import Model, TInput, TTarget

TModel = TypeVar('TModel', bound=Model)

class TrainingContext(Generic[TModel], ABC):
    
    @property
    @abstractmethod
    def model(self) -> TModel:
        pass

    @property
    @abstractmethod
    def epoch(self) -> int:
        pass

    @property
    @abstractmethod
    def iteration(self) -> int:
        pass