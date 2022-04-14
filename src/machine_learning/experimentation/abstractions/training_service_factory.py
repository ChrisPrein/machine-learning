from abc import ABC, abstractmethod
from typing import Any, Dict, Generic
from experimentation.experiment.abstractions.experimentation_service import ExperimentationService, TExperimentSettings, TExperimentResult

from ...evaluation.abstractions.evaluation_context import TModel, EvaluationContext
from ...modeling.abstractions.model import Model, TInput, TTarget
from ...evaluation.abstractions.evaluation_service import EvaluationService
from ...training.abstractions.training_service import TrainingService
from ...training.abstractions.training_context import TrainingContext
from ..machine_learning_experiment_settings import MachineLearningExperimentSettings, TTrainingServiceSettings

class TrainingServiceFactory(Generic[TInput, TTarget, TModel, TTrainingServiceSettings], ABC):

    @abstractmethod
    def create(self, settings: TTrainingServiceSettings) -> TrainingService[TInput, TTarget, TModel, TrainingContext[TModel], EvaluationContext[TInput, TTarget, TModel]]:
        pass