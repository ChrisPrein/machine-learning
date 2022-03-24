from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic
from datetime import timedelta, datetime

from numpy import number

from .training_context import TrainingContext
from ...modeling.abstractions.model import Model

TTrainingContext = TypeVar('TTrainingContext', bound=TrainingContext)

class StopCondition(Generic[TTrainingContext], ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def is_satisfied(self, context: TTrainingContext) -> bool:
        pass