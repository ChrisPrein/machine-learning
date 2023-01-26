from dataclasses import dataclass
from typing import *

from ..modeling.model import Model, TInput, TTarget, TOutput

__all__ = ['Prediction', 'EvaluationContext', 'TModel']

TModel = TypeVar('TModel', bound=Model)

@dataclass(frozen=True)
class Prediction(Generic[TInput, TTarget, TOutput]):
    input: TInput
    prediction: TOutput
    target: TTarget

@dataclass
class EvaluationContext(Generic[TInput, TTarget, TOutput, TModel]):
    model: Optional[TModel]
    dataset_name: str
    predictions: Deque[Prediction[TInput, TTarget, TOutput]]
    current_batch_index: int
    losses: Deque[Union[float, Dict[str, float]]]