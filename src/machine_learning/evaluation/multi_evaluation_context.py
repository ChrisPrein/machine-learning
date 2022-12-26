from abc import ABC
from dataclasses import dataclass
from typing import *

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