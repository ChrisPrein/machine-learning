from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Optional, Dict
from ..modeling.abstractions.model import TTarget, TInput
from .abstractions.evaluation_context import EvaluationContext, TModel

class DefaultEvaluationContext(EvaluationContext[TTarget, TModel], ABC):
    def __init__(self, model: TModel):
        self.__model: TModel = model
        self.__predictions: List[TTarget] = []

    @property
    def model(self) -> TModel:
        return self.__model

    @property
    def predictions(self) -> List[TTarget]:
        return self.__predictions