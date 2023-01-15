from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Any, Callable, Union, overload

__all__ = ['TInput', 'TTarget', 'InputBatch', 'TargetBatch', 'Input', 'Target', 'Model']

TInput = TypeVar('TInput')
TTarget = TypeVar('TTarget')
InputBatch = List[TInput]
TargetBatch = List[TTarget]
Input = Union[TInput, InputBatch[TInput]]
Target = Union[TTarget, TargetBatch[TTarget]]

class Model(Generic[TInput, TTarget], ABC):
    def __init__(self):
        super().__init__()

    @overload
    def predict_step(self, input: TInput) -> TTarget: ...
    @overload
    def predict_step(self, input: InputBatch[TInput]) -> TargetBatch[TTarget]: ...
    @abstractmethod
    def predict_step(self, input: Input[TInput]) -> Target[TTarget]: ...