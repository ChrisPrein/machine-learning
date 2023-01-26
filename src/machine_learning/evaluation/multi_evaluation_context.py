from abc import ABC
from dataclasses import dataclass
from typing import *

from ..modeling.model import TInput, TTarget

from .evaluation_context import *

__all__ = ['Score', 'MultiEvaluationContext']

@dataclass(frozen=True)
class Score():
    value: float
    metric_name: str
    dataset_name: str

@dataclass()
class MultiEvaluationContext(ABC):
    current_dataset_index: int
    scores: Dict[str, Dict[str, Score]]