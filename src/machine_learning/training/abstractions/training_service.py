from abc import ABC, abstractmethod
from typing import Optional, TypeVar, List, Generic, Dict, Tuple, Union
from torch.utils.data.dataset import Dataset
from dataclasses import dataclass

from ...parameter_tuning.abstractions.objective_function import ObjectiveFunction
from ...modeling.abstractions.model import TInput, TTarget
from .stop_condition import TModel, StopCondition, TrainingContext

class TrainingService(Generic[TInput, TTarget, TModel], ABC):
    
    @abstractmethod
    async def train(self, model: TModel, dataset: Union[Tuple[str, Dataset[Tuple[TInput, TTarget]]], Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]], objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget, TModel]], primary_objective: Optional[str] = None, validation_dataset: Optional[Dataset[Tuple[TInput, TTarget]]] = None) -> TModel:
        pass

    @abstractmethod
    async def train_on_multiple_datasets(self, model: TModel, datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]], objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget, TModel]], primary_objective: Optional[str] = None, validation_dataset: Optional[Dataset[Tuple[TInput, TTarget]]] = None) -> TModel:
        pass