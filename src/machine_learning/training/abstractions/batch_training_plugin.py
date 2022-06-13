from abc import ABC, abstractmethod
from logging import Logger
from typing import Generic, Tuple
from torch.utils.data import Dataset, random_split

from .stop_condition import TModel, StopCondition, TrainingContext
from ...modeling.abstractions.model import Model, TInput, TTarget


class BatchTrainingPlugin(Generic[TInput, TTarget, TModel], ABC):
    pass

class PreMultiLoop(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_multi_loop(self, logger: Logger):
        pass

class PostMultiLoop(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_multi_loop(self, logger: Logger):
        pass

class PreLoop(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_loop(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        pass

class PostLoop(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_loop(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        pass

class PreEpoch(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_epoch(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel], validation_dataset: Dataset[Tuple[TInput, TTarget]]):
        pass

class PostEpoch(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_epoch(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel], validation_dataset: Dataset[Tuple[TInput, TTarget]]):
        pass

class PreTrain(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_train(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        pass

class PostTrain(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_train(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        pass

