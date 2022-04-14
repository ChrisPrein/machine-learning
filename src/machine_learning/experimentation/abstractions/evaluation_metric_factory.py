from abc import ABC, abstractmethod
from typing import Any, Dict, Generic
from experimentation.experiment.abstractions.experimentation_service import ExperimentationService, TExperimentSettings, TExperimentResult

from ...evaluation.abstractions.evaluation_context import TModel, EvaluationContext
from ...modeling.abstractions.model import Model, TInput, TTarget
from ...evaluation.abstractions.evaluation_service import EvaluationService
from ...evaluation.abstractions.evaluation_metric import EvaluationMetric
from ..machine_learning_experiment_settings import MachineLearningExperimentSettings, TEvaluationMetricSettings

class EvaluationMetricFactory(Generic[TInput, TTarget, TModel, TEvaluationMetricSettings], ABC):

    @abstractmethod
    def create(self, settings: TEvaluationMetricSettings) -> Dict[str, EvaluationMetric[EvaluationContext[TInput, TTarget, TModel]]]:
        pass