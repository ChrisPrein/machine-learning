from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Tuple
from torch.utils.data import Dataset, random_split
from experimentation.experiment.abstractions.experimentation_service import ExperimentationService, TExperimentSettings, TExperimentResult

from ...evaluation.abstractions.evaluation_context import TModel
from ...modeling.abstractions.model import Model, TInput, TTarget


class DatasetFactoryFactory(Generic[TInput, TTarget, TExperimentSettings], ABC):

    @abstractmethod
    def create(self, settings: TExperimentSettings) -> Dataset[Tuple[TInput, TTarget]]:
        pass