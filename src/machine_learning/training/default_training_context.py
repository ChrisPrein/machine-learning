from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Optional, Dict

from .abstractions.training_context import TrainingContext, TModel
from ..modeling.abstractions.model import TTarget, TInput

class DefaultTrainingContext(TrainingContext[TModel], ABC):
    def __init__(self, model: TModel):
        self.__model: TModel = model
        self.__epoch: int = 0
        self.__iteration: int = 0

    @property
    def model(self) -> TModel:
        return self.__model

    @property
    def epoch(self) -> int:
        return self.__epoch

    @property
    def iteration(self) -> int:
        return self.__iteration

    @model.setter
    def model(self, value: TModel):
        self.__model = value

    @epoch.setter
    def epoch(self, value: int):
        self.__epoch = value

    @iteration.setter
    def iteration(self, value: int):
        self.__iteration = value