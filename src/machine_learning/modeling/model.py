from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, List, Any, Callable, Union, overload

__all__ = ['TInput', 'TTarget', 'TOutput', 'InputBatch', 'OutputBatch', 'Input', 'Output', 'Model']

TInput = TypeVar('TInput')
TTarget = TypeVar('TTarget')
TAuxiliary = TypeVar('TAuxiliary')

@dataclass
class ModelOuput(Generic[TTarget, TAuxiliary]):
    prediction: TTarget
    auxiliary_output: TAuxiliary

TOutput = TypeVar('TOutput', bound=ModelOuput)

InputBatch = List[TInput]
OutputBatch = List[TOutput]

Input = Union[TInput, InputBatch[TInput]]
Output = Union[TOutput, OutputBatch[TOutput]]


class Model(Generic[TInput, TOutput], ABC):
    def __init__(self):
        super().__init__()

    @overload
    def predict_step(self, input: TInput) -> TOutput: ...
    @overload
    def predict_step(self, input: InputBatch[TInput]) -> OutputBatch[TOutput]: ...
    @abstractmethod
    def predict_step(self, input: Input[TInput]) -> Output[TOutput]: ...