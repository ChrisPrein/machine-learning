from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Callable, Dict, Optional

__all__ = ['TuningService', 'TModel', 'Dataset', 'TrainingDataset']

class TuningService(ABC):
    @abstractmethod
    async def tune(self, training_function: Callable[[Dict[str, Any]], None], params: Dict[str, Any], logger: Optional[Logger] = None) -> None: ...