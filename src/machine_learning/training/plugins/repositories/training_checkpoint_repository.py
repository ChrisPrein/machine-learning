from abc import ABC, abstractmethod
from dataclasses import dataclass

__all__ = ['TrainingCheckpointRepository', 'TrainingCheckpoint']

@dataclass
class TrainingCheckpoint:
    current_epoch: int
    current_batch_index: int
    continue_training: bool

class TrainingCheckpointRepository(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    async def get(self, name: str) -> TrainingCheckpoint: ...

    @abstractmethod
    async def save(self, checkpoint: TrainingCheckpoint, name: str): ...