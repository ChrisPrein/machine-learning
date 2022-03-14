from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Optional, Dict
from datetime import timedelta, datetime
from .abstractions.evaluation_context import *
from .abstractions.evaluation_metric import *
from .abstractions.evaluation_service import *

class DefaultEvaluationContext(EvaluationContext[TModel], ABC):
    def __init__(self, model: TModel):
        self.__model: TModel = model
        self.__predictions: List[TTarget] = []

    @property
    def model(self) -> TModel:
        return self.__model

    @property
    def predictions(self) -> List[TTarget]:
        return self.__predictions