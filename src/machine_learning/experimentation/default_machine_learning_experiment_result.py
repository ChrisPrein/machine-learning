from dataclasses import dataclass
from typing import Dict, Generic
from experimentation.experiment.abstractions.experiment_result import ExperimentResult

from ..modeling.abstractions.model import TInput, TTarget
from ..evaluation.abstractions.evaluation_context import TModel

@dataclass
class DefaultMachineLearningExperimentResult(Generic[TModel], ExperimentResult):
    model: TModel
    scores: Dict[str, float]