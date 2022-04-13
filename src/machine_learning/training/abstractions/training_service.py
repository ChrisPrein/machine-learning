from abc import ABC, abstractmethod
from typing import Optional, TypeVar, List, Generic, Dict, Tuple
from dataset_handling.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from ...parameter_tuning.abstractions.objective_function import ObjectiveFunction, TEvaluationContext
from .stop_condition import StopCondition, TTrainingContext
from ...modeling.abstractions.model import TInput, TTarget
from .training_context import TModel


class TrainingService(Generic[TInput, TTarget, TModel, TTrainingContext, TEvaluationContext], ABC):
    
    @abstractmethod
    async def train(self, model: TModel, dataset: Dataset[Tuple[TInput, TTarget]], stop_conditions: Dict[str, StopCondition[TTrainingContext]], objective_functions: Dict[str, ObjectiveFunction[TEvaluationContext]], primary_objective: Optional[str] = None) -> TModel:
        pass

    @abstractmethod
    async def train(self, model: TModel, datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TTrainingContext]], objective_functions: Dict[str, ObjectiveFunction[TEvaluationContext]], primary_objective: Optional[str] = None) -> TModel:
        pass