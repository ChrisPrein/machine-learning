from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from typing import Dict, Generic, List, Tuple, Deque, TypeVar
from torch.utils.data import Dataset, random_split

from ...modeling.abstractions.model import Model, TInput, TTarget

TModel = TypeVar('TModel', bound=Model)

@dataclass(frozen=True)
class Prediction(Generic[TInput, TTarget]):
    input: TInput
    prediction: TTarget
    target: TTarget

@dataclass(frozen=True)
class EvaluationContext(Generic[TInput, TTarget, TModel], ABC):
    model: TModel
    dataset_name: str
    predictions: Deque[Prediction[TInput, TTarget]]
    current_batch_index: int