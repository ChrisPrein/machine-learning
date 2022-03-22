from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Optional, Dict

from .abstractions.training_context import TrainingContext, TModel, Score
from ..modeling.abstractions.model import TTarget, TInput

class DefaultTrainingContext(TrainingContext[TModel], ABC):
    def __init__(self, model: TModel, objectives: List[str], primary_objective: str):
        self.__model: TModel = model
        self.__current_epoch: int = 0
        self.__current_iteration: int = 0
        self.__primary_objective: str = primary_objective
        self.__scores: Dict[str, List[Score]] = {objective: [] for objective in objectives}

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
    def primary_scores(self) -> List[Score]:
        return self.__scores[self.__primary_objective]

    @property
    @abstractmethod
    def scores(self) -> Dict[str, List[Score]]:
        return self.__scores

    @model.setter
    def model(self, value: TModel):
        self.__model = value

    @current_epoch.setter
    def current_epoch(self, value: int):
        self.__current_epoch = value

    @current_iteration.setter
    def current_iteration(self, value: int):
        self.__current_iteration = value