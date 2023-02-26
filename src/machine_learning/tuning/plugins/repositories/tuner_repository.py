from abc import ABC, abstractmethod
from dataclasses import dataclass
from ....evaluation import EvaluationResult
from ray.tune import Tuner

__all__ = ['TunerRepository']

class TunerRepository(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    async def get(self, name: str) -> Tuner: ...