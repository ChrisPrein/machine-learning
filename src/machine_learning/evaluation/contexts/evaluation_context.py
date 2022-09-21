from dataclasses import dataclass
from typing import *

from ...modeling.abstractions.model import *

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