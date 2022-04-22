from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, List, Generic, Dict

from ...modeling.abstractions.model import Model
from ...modeling.abstractions.model import Model, TInput, TTarget
from ...parameter_tuning.abstractions.objective_function import OptimizationType

TModel = TypeVar('TModel', bound=Model)

@dataclass(frozen=True)
class Score:
    epoch: int
    iteration: int
    score: float
    optimization_type: OptimizationType

@dataclass
class TrainingContext(Generic[TModel]):
    model: TModel
    current_epoch: int
    current_iteration: int
    scores: Dict[str, List[Score]]
    _primary_objective: str

    @property
    @abstractmethod
    def primary_scores(self) -> List[Score]:
        return self.scores[self._primary_objective]

class StopCondition(Generic[TModel], ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def is_satisfied(self, context: TrainingContext[TModel]) -> bool:
        pass