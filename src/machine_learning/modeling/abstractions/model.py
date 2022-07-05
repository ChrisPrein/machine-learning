from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Dict, Any, Callable, Union

TInput = TypeVar('TInput')
TTarget = TypeVar('TTarget')

class Model(Generic[TInput, TTarget], ABC):

    @abstractmethod
    def predict_step(self, batch: List[TInput]) -> List[TTarget]: 
        pass

    @abstractmethod
    def training_step(self, input_batch: List[TInput], target_batch: List[TTarget]) -> Union[float, Dict[str, float]]:
        pass

    __call__ : Callable[..., Any] = predict_step