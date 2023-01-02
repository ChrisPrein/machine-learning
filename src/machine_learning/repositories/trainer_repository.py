from abc import ABC, abstractmethod
from typing import Generic
from ..training import TTrainer

__all__ = ['TrainerRepository']

class TrainerRepository(Generic[TTrainer], ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    async def get(self, name: str) -> TTrainer: ...

    @abstractmethod
    async def save(self, trainer: TTrainer, name: str): ...