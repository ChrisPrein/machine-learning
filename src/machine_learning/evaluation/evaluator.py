from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Tuple, Union, overload

from ..modeling.model import TInput, TTarget, TOutput
from .evaluation_service import TModel

__all__ = ['InputBatch', 'Input', 'TargetBatch', 'Target', 'EvaluatorResult', 'Evaluator']

InputBatch = Iterable[TInput]
Input = Union[TInput, InputBatch[TInput]]
TargetBatch = Iterable[TTarget]
Target = Union[TTarget, TargetBatch[TTarget]]
EvaluatorResult = Tuple[List[TOutput], Union[float, Dict[str, float]]]

class Evaluator(Generic[TInput, TTarget, TOutput, TModel], ABC):
    def __init__(self):
        super().__init__()

    @overload
    def evaluation_step(self, model: TModel, input: TInput, target: TTarget, logger: Optional[Logger] = None) -> EvaluatorResult[TOutput]: ...
    @overload
    def evaluation_step(self, model: TModel, input: InputBatch[TInput], target: TargetBatch[TTarget], logger: Optional[Logger] = None) -> EvaluatorResult[TOutput]: ...
    @abstractmethod
    def evaluation_step(self, model: TModel, input: Input[TInput], target: Target[TTarget], logger: Optional[Logger] = None) -> EvaluatorResult[TOutput]: ...