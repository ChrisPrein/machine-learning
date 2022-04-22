from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Callable

from ..evaluation.abstractions.evaluation_metric import TModel
from .abstractions.stop_condition import StopCondition, TrainingContext

class CustomStopCondition(StopCondition[TModel], ABC):
    def __init__(self, expression: Callable[[TrainingContext[TModel]], bool]):
        if expression is None:
            raise ValueError("expression can't be empty")

        self.expression: Callable[[TrainingContext[TModel]], bool] = expression

    def reset(self):
        pass

    def is_satisfied(self, context: TrainingContext[TModel]) -> bool:
        return self.expression(context)