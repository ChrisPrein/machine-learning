from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic
from ...modeling.abstractions.model import Model, TInput, TTarget

TModel = TypeVar('TModel', bound=Model)

class EvaluationContext(Generic[TTarget, TModel], ABC):
    
    @property
    @abstractmethod
    def model(self) -> TModel:
        pass

    @property
    @abstractmethod
    def predictions(self) -> List[TTarget]:
        pass