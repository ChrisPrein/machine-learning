from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from typing import Dict, Generic, List, Tuple, Deque, TypeVar
from torch.utils.data import Dataset, random_split

from ...modeling.abstractions.model import Model, TInput, TTarget
from .evaluation_context import *

@dataclass(frozen=True)
class Score():
    value: float
    metric_name: str
    dataset_name: str

@dataclass()
class MultiEvaluationContext(Generic[TInput, TTarget, TModel], ABC):
    current_dataset_index: int
    scores: Dict[str, Dict[str, Score]]