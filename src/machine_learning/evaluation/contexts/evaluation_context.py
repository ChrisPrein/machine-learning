from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from typing import Dict, Generic, List, Optional, Tuple, Deque, TypeVar, Union
from torch.utils.data import Dataset, random_split

from ...modeling.abstractions.model import Model, TInput, TTarget

TModel = TypeVar('TModel', bound=Model)

@dataclass(frozen=True)
class Prediction(Generic[TInput, TTarget]):
    input: TInput
    prediction: TTarget
    target: TTarget

@dataclass
class EvaluationContext(Generic[TInput, TTarget, TModel]):
    model: Optional[TModel]
    dataset_name: str
    predictions: Deque[Prediction[TInput, TTarget]]
    current_batch_index: int
    losses: Deque[Union[float, Dict[str, float]]]