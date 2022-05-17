from abc import ABC, abstractmethod
from dataclasses import dataclass
import enum
from typing import TypeVar, List, Generic, Dict

from machine_learning.evaluation.abstractions.evaluation_metric import Prediction

from ...modeling.abstractions.model import Model
from ...modeling.abstractions.model import Model, TInput, TTarget
from ...parameter_tuning.abstractions.objective_function import OptimizationType

TModel = TypeVar('TModel', bound=Model)

@dataclass(frozen=True)
class Score(Generic[TInput, TTarget]):
    epoch: int
    iteration: int
    score: float
    optimization_type: OptimizationType
    prediction: Prediction[TInput, TTarget]

@dataclass
class TrainingContext(Generic[TInput, TTarget, TModel]):
    model: TModel
    current_epoch: int
    current_iteration: int
    scores: Dict[str, List[Score[TInput, TTarget]]]
    _primary_objective: str

    @property
    def primary_scores(self) -> List[Score[TInput, TTarget]]:
        return self.scores[self._primary_objective]

    @property
    def current_scores(self) -> Dict[str, List[Score[TInput, TTarget]]]:
        filtered_scores: Dict[str, List[Score[TInput, TTarget]]] = {}

        for score_name, scores in self.scores.items():
            filtered_scores[score_name] = list(filter(lambda score: score.epoch == self.current_epoch and score.iteration == self.current_iteration, scores))

        return filtered_scores

class StopCondition(Generic[TInput, TTarget, TModel], ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def is_satisfied(self, context: TrainingContext[TInput, TTarget, TModel]) -> bool:
        pass