from abc import ABC, abstractmethod
from typing import Generic
from ..evaluation import TModel

__all__ = ['ModelRepository']

class ModelRepository(Generic[TModel], ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    async def get(self, name: str) -> TModel: ...

    @abstractmethod
    async def save(self, model: TModel, name: str): ...