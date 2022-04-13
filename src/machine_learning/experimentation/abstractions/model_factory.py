from abc import ABC, abstractmethod
from typing import Any, Dict, Generic
from experimentation.experiment.abstractions.experimentation_service import ExperimentationService, TExperimentSettings, TExperimentResult

from ...evaluation.abstractions.evaluation_context import TModel

class ModelFactory(Generic[TModel, TExperimentSettings], ABC):

    @abstractmethod
    def create(self, settings: TExperimentSettings) -> TModel:
        pass