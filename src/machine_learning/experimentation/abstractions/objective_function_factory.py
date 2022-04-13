from abc import ABC, abstractmethod
from typing import Any, Dict, Generic
from experimentation.experiment.abstractions.experimentation_service import ExperimentationService, TExperimentSettings, TExperimentResult

from ...evaluation.abstractions.evaluation_context import TModel, EvaluationContext
from ...modeling.abstractions.model import Model, TInput, TTarget
from ...evaluation.abstractions.evaluation_service import EvaluationService
from ...evaluation.abstractions.evaluation_metric import EvaluationMetric
from ...parameter_tuning.abstractions.objective_function import ObjectiveFunction

class ObjectiveFunctionFactory(Generic[TInput, TTarget, TModel, TExperimentSettings], ABC):

    @abstractmethod
    def create(self, settings: TExperimentSettings) -> Dict[str, ObjectiveFunction[EvaluationContext[TInput, TTarget, TModel]]]:
        pass