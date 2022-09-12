from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Dict, Any, Callable, Union, Tuple

TInput = TypeVar('TInput')
TTarget = TypeVar('TTarget')

class Model(Generic[TInput, TTarget], ABC):

    @abstractmethod
    def predict_step(self, batch: List[TInput]) -> List[TTarget]: ...

    @abstractmethod
    def evaluation_step(self, input_batch: List[TInput], target_batch: List[TTarget]) -> Tuple[List[TTarget], Union[float, Dict[str, float]]]: ...

    @abstractmethod
    def training_step(self, input_batch: List[TInput], target_batch: List[TTarget]) -> Union[float, Dict[str, float]]: ...

    __call__ : Callable[..., Any] = predict_step