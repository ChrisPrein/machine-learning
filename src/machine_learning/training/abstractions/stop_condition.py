from abc import ABC, abstractmethod
from dataclasses import dataclass
import enum
from typing import TypeVar, List, Generic, Dict

from ...modeling.abstractions.model import Model
from ...modeling.abstractions.model import Model, TInput, TTarget
from ...parameter_tuning.abstractions.objective_function import OptimizationType
from ...evaluation.abstractions.evaluation_service import Score
from .batch_training_plugin import TrainingContext

TModel = TypeVar('TModel', bound=Model)

class StopCondition(Generic[TInput, TTarget, TModel], ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def is_satisfied(self, context: TrainingContext[TInput, TTarget, TModel]) -> bool:
        pass