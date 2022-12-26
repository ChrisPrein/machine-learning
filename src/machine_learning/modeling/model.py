from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Dict, Any, Callable, Union, Tuple, overload

TInput = TypeVar('TInput')
TTarget = TypeVar('TTarget')
INPUT_BATCH = List[TInput]
TARGET_BATCH = List[TTarget]
INPUT = Union[TInput, INPUT_BATCH]
TARGET = Union[TTarget, TARGET_BATCH]

class Model(Generic[TInput, TTarget], ABC):
    def __init__(self):
        super().__init__()

    @overload
    def predict_step(self, input: TInput) -> TTarget: ...
    @overload
    def predict_step(self, input: INPUT_BATCH) -> TARGET_BATCH: ...
    @abstractmethod
    def predict_step(self, input: INPUT) -> TARGET: ...

    __call__ : Callable[..., Any] = predict_step