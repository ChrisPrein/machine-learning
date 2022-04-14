from abc import ABC, abstractmethod
from typing import Any, Dict, Generic
from experimentation.experiment.abstractions.experimentation_service import ExperimentationService, TExperimentSettings, TExperimentResult

from ...evaluation.abstractions.evaluation_context import TModel
from ..machine_learning_experiment_settings import MachineLearningExperimentSettings, TModelSettings

class ModelFactory(Generic[TModel, TModelSettings], ABC):

    @abstractmethod
    def create(self, settings: TModelSettings) -> TModel:
        pass