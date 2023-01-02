from abc import ABC, abstractmethod
from ..training.plugins.checkpoint_plugin import TrainingCheckpoint

__all__ = ['TrainingCheckpointRepository']

class TrainingCheckpointRepository(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    async def get(self, name: str) -> TrainingCheckpoint: ...

    @abstractmethod
    async def save(self, checkpoint: TrainingCheckpoint, name: str): ...