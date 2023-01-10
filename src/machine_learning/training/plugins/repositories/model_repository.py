from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from ....modeling import Model

__all__ = ['ModelRepository']

TModel = TypeVar('TModel', bound=Model)

class ModelRepository(Generic[TModel], ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    async def get(self, name: str) -> TModel: ...

    @abstractmethod
    async def save(self, model: TModel, name: str): ...