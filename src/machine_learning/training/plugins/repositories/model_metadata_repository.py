from abc import ABC, abstractmethod
from dataclasses import dataclass

__all__ = ['ModelMetadataRepository', 'ModelMetadata']

@dataclass
class ModelMetadata:
    loss: float

class ModelMetadataRepository(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    async def get(self, name: str) -> ModelMetadata: ...

    @abstractmethod
    async def save(self, metadata: ModelMetadata, name: str): ...