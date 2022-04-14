from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Tuple, TypeVar
from torch.utils.data import Dataset, random_split
from experimentation.experiment.abstractions.experimentation_service import ExperimentationService, TExperimentSettings, TExperimentResult

from ...evaluation.abstractions.evaluation_context import TModel
from ...modeling.abstractions.model import Model, TInput, TTarget
from ..machine_learning_experiment_settings import MachineLearningExperimentSettings

TDatasetSettings = TypeVar("TDatasetSettings")

class DatasetFactory(Generic[TInput, TTarget, TDatasetSettings], ABC):

    @abstractmethod
    def create(self, settings: TDatasetSettings) -> Dict[str, Dataset[Tuple[TInput, TTarget]]]:
        pass