from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic
from datetime import timedelta, datetime
from enum import Enum
from ...modeling.abstractions.model import *

TModel = TypeVar('TModel', Model[TInput, TTarget])

class EvaluationContext(Generic[TTarget, TModel], ABC):
    
    @property
    @abstractmethod
    def model(self) -> TModel:
        pass

    @property
    @abstractmethod
    def predictions(self) -> List[TTarget]:
        pass