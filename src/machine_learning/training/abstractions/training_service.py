from abc import ABC, abstractmethod
from typing import Optional, TypeVar, List, Generic, Dict, Tuple
from torch.utils.data.dataset import Dataset

from ...parameter_tuning.abstractions.objective_function import ObjectiveFunction
from ...modeling.abstractions.model import TInput, TTarget
from .stop_condition import TModel, StopCondition


class TrainingService(Generic[TInput, TTarget, TModel], ABC):
    
    @abstractmethod
    async def train(self, model: TModel, dataset: Dataset[Tuple[TInput, TTarget]], stop_conditions: Dict[str, StopCondition[TModel]], objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget, TModel]], primary_objective: Optional[str] = None) -> TModel:
        pass

    @abstractmethod
    async def train(self, model: TModel, datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TModel]], objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget, TModel]], primary_objective: Optional[str] = None) -> TModel:
        pass