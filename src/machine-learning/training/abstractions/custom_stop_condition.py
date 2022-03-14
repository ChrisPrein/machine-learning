from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Callable
from datetime import timedelta, datetime

from numpy import number
from .stop_condition import StopCondition, TModel

class CustomStopCondition(StopCondition[TModel], ABC):
    def __init__(self, expression: Callable[[TModel, number, number], bool]):
        if expression is None:
            raise ValueError("expression can't be empty")

        self.expression: Callable[[TModel, number, number], bool] = expression

    def is_satisfied(self, model: TModel, epoch: number, iteration: number) -> bool:
        return self.expression(model, epoch, iteration)