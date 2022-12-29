from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Tuple, Union, overload

from ..modeling.model import TInput, TTarget
from .evaluation_service import TModel

__all__ = ['InputBatch', 'Input', 'TargetBatch', 'Target', 'EvaluatorResult']

InputBatch = Iterable[TInput]
Input = Union[TInput, InputBatch[TInput]]
TargetBatch = Iterable[TTarget]
Target = Union[TInput, TargetBatch[TTarget]]
EvaluatorResult = Tuple[List[TTarget], Union[float, Dict[str, float]]]

class Evaluator(Generic[TInput, TTarget, TModel], ABC):
    def __init__(self):
        super().__init__()

    @overload
    def evaluation_step(self, model: TModel, input: TInput, target: TTarget, logger: Optional[Logger] = None) -> EvaluatorResult[TTarget]: ...
    @overload
    def evaluation_step(self, model: TModel, input: InputBatch[TInput], target: TargetBatch[TTarget], logger: Optional[Logger] = None) -> EvaluatorResult[TTarget]: ...
    @abstractmethod
    def evaluation_step(self, model: TModel, input: Input[TInput], target: Target[TTarget], logger: Optional[Logger] = None) -> EvaluatorResult[TTarget]: ...

    __call__ : Callable[..., Any] = evaluation_step