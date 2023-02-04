from abc import ABC, abstractmethod
from dataclasses import dataclass
from ....evaluation import EvaluationResult
from ...batch_training_service import Loss

__all__ = ['ModelMetadataRepository', 'ModelMetadata']

@dataclass
class ModelMetadata:
    loss: Loss
    performance: EvaluationResult

class ModelMetadataRepository(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    async def get(self, name: str) -> ModelMetadata: ...

    @abstractmethod
    async def save(self, metadata: ModelMetadata, name: str): ...