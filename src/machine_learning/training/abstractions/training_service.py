from abc import ABC, abstractmethod
import asyncio
from logging import Logger
from optparse import Option
from typing import Optional, TypeVar, List, Generic, Dict, Tuple, Union, overload
from uuid import UUID
from torch.utils.data.dataset import Dataset

from ...modeling.abstractions.model import TInput, TTarget
from ...evaluation.abstractions.evaluation_metric import *
from .stop_condition import TModel, StopCondition, TrainingContext



class TrainingService(Generic[TInput, TTarget, TModel], ABC):
    
    @overload
    async def train(self, model: TModel, dataset: Dataset[Tuple[TInput, TTarget]], stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]], primary_objective: Optional[str] = None, validation_dataset: Optional[Dataset[Tuple[TInput, TTarget]]] = None, logger: Optional[Logger] = None) -> TModel: ...
    @overload
    async def train(self, model: TModel, dataset: Tuple[str, Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]], primary_objective: Optional[str] = None, validation_dataset: Optional[Tuple[str, Dataset[Tuple[TInput, TTarget]]]] = None, logger: Optional[Logger] = None) -> TModel: ...
    @abstractmethod
    async def train(self, model: TModel, dataset: Union[Tuple[str, Dataset[Tuple[TInput, TTarget]]], Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]], primary_objective: Optional[str] = None, validation_dataset: Optional[Union[Tuple[str, Dataset[Tuple[TInput, TTarget]]], Dataset[Tuple[TInput, TTarget]]]] = None, logger: Optional[Logger] = None) -> TModel: ...

    @overload
    async def train_on_multiple_datasets(self, model: TModel, datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]], primary_objective: Optional[str] = None, validation_dataset: Optional[Tuple[str, Dataset[Tuple[TInput, TTarget]]]] = None) -> TModel: ...
    @overload
    async def train_on_multiple_datasets(self, model: TModel, datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]], primary_objective: Optional[str] = None, validation_dataset: Optional[Dataset[Tuple[TInput, TTarget]]] = None) -> TModel: ...
    @abstractmethod
    async def train_on_multiple_datasets(self, model: TModel, datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]], primary_objective: Optional[str] = None, validation_dataset: Optional[Union[Tuple[str, Dataset[Tuple[TInput, TTarget]]], Dataset[Tuple[TInput, TTarget]]]] = None) -> TModel: ...