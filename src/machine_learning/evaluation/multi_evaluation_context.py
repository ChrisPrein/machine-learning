from abc import ABC
from dataclasses import dataclass
from typing import *

from ..modeling.model import TInput, TTarget

from .evaluation_context import *

__all__ = ['MultiEvaluationContext']

@dataclass()
class MultiEvaluationContext(ABC):
    current_dataset_index: int
    scores: Dict[str, Dict[str, float]]