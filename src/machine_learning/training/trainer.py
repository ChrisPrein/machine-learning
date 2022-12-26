from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Tuple, Union, overload

from ..modeling.model import TInput, TTarget
from .training_service import TModel

INPUT_BATCH = Iterable[TInput]
INPUT = Union[TInput, INPUT_BATCH]
TARGET_BATCH = Iterable[TTarget]
TARGET = Union[TInput, TARGET_BATCH]
TRAINER_RESULT = Tuple[List[TTarget], Union[float, Dict[str, float]]]

class Trainer(Generic[TInput, TTarget, TModel], ABC):
    def __init__(self):
        super().__init__()

    @overload
    def train_step(self, model: TModel, input: TInput, target: TTarget, logger: Optional[Logger] = None) -> TRAINER_RESULT: ...
    @overload
    def train_step(self, model: TModel, input: INPUT_BATCH, target: TARGET_BATCH, logger: Optional[Logger] = None) -> TRAINER_RESULT: ...
    @abstractmethod
    def train_step(self, model: TModel, input: INPUT, target: TARGET, logger: Optional[Logger] = None) -> TRAINER_RESULT: ...

    __call__ : Callable[..., Any] = train_step