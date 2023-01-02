from abc import ABC, abstractmethod
from ..training.plugins.model_store_plugin import ModelMetadata

__all__ = ['ModelMetadataRepository']

class ModelMetadataRepository(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    async def get(self, name: str) -> ModelMetadata: ...

    @abstractmethod
    async def save(self, metadata: ModelMetadata, name: str): ...