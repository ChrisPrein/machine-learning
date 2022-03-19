from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, List, Generic
from ...modeling.abstractions.model import Model, TInput, TTarget

TModel = TypeVar('TModel', bound=Model)

@dataclass
class Prediction(Generic[TInput, TTarget]):
    input: TInput
    prediction: TTarget
    target: TTarget

class EvaluationContext(Generic[TInput, TTarget, TModel], ABC):
    
    @property
    @abstractmethod
    def model(self) -> TModel:
        pass

    @property
    @abstractmethod
    def predictions(self) -> List[Prediction[TInput, TTarget]]:
        pass