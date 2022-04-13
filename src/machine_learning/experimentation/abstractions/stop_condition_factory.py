from abc import ABC, abstractmethod
from typing import Any, Dict, Generic
from experimentation.experiment.abstractions.experimentation_service import ExperimentationService, TExperimentSettings, TExperimentResult

from ...evaluation.abstractions.evaluation_context import TModel, EvaluationContext
from ...modeling.abstractions.model import Model, TInput, TTarget
from ...evaluation.abstractions.evaluation_service import EvaluationService
from ...evaluation.abstractions.evaluation_metric import EvaluationMetric
from ...parameter_tuning.abstractions.objective_function import ObjectiveFunction
from ...training.abstractions.stop_condition import StopCondition
from ...training.abstractions.training_context import TrainingContext

class StopConditionFactory(Generic[TModel, TExperimentSettings], ABC):

    @abstractmethod
    def create(self, settings: TExperimentSettings) -> Dict[str, StopCondition[TrainingContext[TModel]]]:
        pass