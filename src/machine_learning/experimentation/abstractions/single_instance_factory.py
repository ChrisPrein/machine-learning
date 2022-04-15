from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar

TSettings = TypeVar('TSettings')
TInstance = TypeVar('TInstance')

class SingleInstanceFactory(Generic[TSettings, TInstance], ABC):

    @abstractmethod
    def create(self, settings: TSettings) -> TInstance:
        pass