from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, List, Generic, Dict
from ...modeling.abstractions.model import Model, TInput, TTarget
from ...parameter_tuning.abstractions.objective_function import OptimizationType

TModel = TypeVar('TModel', bound=Model)

@dataclass
class Score:
    epoch: int
    iteration: int
    score: float
    optimization_type: OptimizationType

class TrainingContext(Generic[TModel], ABC):
    
    @property
    @abstractmethod
    def model(self) -> TModel:
        pass

    @property
    @abstractmethod
    def current_epoch(self) -> int:
        pass

    @property
    @abstractmethod
    def current_iteration(self) -> int:
        pass

    @property
    @abstractmethod
    def primary_scores(self) -> List[Score]:
        pass

    @property
    @abstractmethod
    def scores(self) -> List[Dict[str, Score]]:
        pass