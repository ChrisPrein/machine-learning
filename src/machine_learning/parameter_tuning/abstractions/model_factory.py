from abc import ABC, abstractmethod
from typing import Any, Dict, Generic

from ...evaluation.abstractions.evaluation_context import TModel

class ModelFactory(Generic[TModel], ABC):

    @abstractmethod
    def create(self, params: Dict[str, Any]) -> TModel:
        pass