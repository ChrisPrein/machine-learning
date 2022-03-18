from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic
from datetime import timedelta, datetime

from numpy import number
from ...modeling.abstractions.model import Model

TModel = TypeVar('TModel', Model)

class StopCondition(Generic[TModel], ABC):
    @abstractmethod
    def is_satisfied(self, model: TModel, epoch: number, iteration: number) -> bool:
        pass