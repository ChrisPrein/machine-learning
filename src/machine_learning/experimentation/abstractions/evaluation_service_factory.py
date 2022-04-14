from abc import ABC, abstractmethod
from typing import Any, Dict, Generic
from experimentation.experiment.abstractions.experimentation_service import ExperimentationService, TExperimentSettings, TExperimentResult

from ...evaluation.abstractions.evaluation_context import TModel, EvaluationContext
from ...modeling.abstractions.model import Model, TInput, TTarget
from ...evaluation.abstractions.evaluation_service import EvaluationService
from ..machine_learning_experiment_settings import MachineLearningExperimentSettings, TEvaluationServiceSettings

class EvaluationServiceFactory(Generic[TInput, TTarget, TModel, TEvaluationServiceSettings], ABC):

    @abstractmethod
    def create(self, settings: TEvaluationServiceSettings) -> EvaluationService[TInput, TTarget, TModel, EvaluationContext[TInput, TTarget, TModel]]:
        pass