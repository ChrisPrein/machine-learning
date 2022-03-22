from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Optional, Dict

from .abstractions.training_context import TrainingContext, TModel, Loss
from ..modeling.abstractions.model import TTarget, TInput

class DefaultTrainingContext(TrainingContext[TModel], ABC):
    def __init__(self, model: TModel):
        self.__model: TModel = model
        self.__current_epoch: int = 0
        self.__current_iteration: int = 0
        self.__loss: List[Loss] = []

    @property
    def model(self) -> TModel:
        return self.__model

    @property
    def current_epoch(self) -> int:
        return self.__current_epoch

    @property
    def current_iteration(self) -> int:
        return self.__current_iteration

    @property
    @abstractmethod
    def loss(self) -> List[Loss]:
        return self.__loss

    @model.setter
    def model(self, value: TModel):
        self.__model = value

    @current_epoch.setter
    def current_epoch(self, value: int):
        self.__current_epoch = value

    @current_iteration.setter
    def current_iteration(self, value: int):
        self.__current_iteration = value