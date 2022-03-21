from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic, Dict, Tuple
from dataset_handling.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from .stop_condition import StopCondition, TTrainingContext
from ...modeling.abstractions.model import TInput, TTarget
from .training_context import TModel


class TrainingService(Generic[TInput, TTarget, TModel, TTrainingContext], ABC):
    
    @abstractmethod
    async def train(self, model: TModel, dataset: Dataset[Tuple[TInput, TTarget]], stop_conditions: Dict[str, StopCondition[TTrainingContext]]) -> TModel:
        pass