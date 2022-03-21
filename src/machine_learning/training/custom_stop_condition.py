from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Callable
from datetime import timedelta, datetime

from numpy import number

from .abstractions.training_context import TrainingContext
from .abstractions.stop_condition import StopCondition, TTrainingContext

class CustomStopCondition(StopCondition[TTrainingContext], ABC):
    def __init__(self, expression: Callable[[TTrainingContext], bool]):
        if expression is None:
            raise ValueError("expression can't be empty")

        self.expression: Callable[[TTrainingContext], bool] = expression

    def reset(self):
        pass

    def is_satisfied(self, context: TrainingContext) -> bool:
        return self.expression(context)