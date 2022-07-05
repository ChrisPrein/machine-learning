from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from typing import Dict, Generic, List, Tuple, Deque, Union
from torch.utils.data import Dataset, random_split

from ...evaluation.abstractions.evaluation_service import Score

from ...modeling.abstractions.model import Model, TInput, TTarget
from ...evaluation.contexts.evaluation_context import *

@dataclass
class TrainingContext(Generic[TInput, TTarget, TModel]):
    model: TModel
    dataset_name: str
    current_epoch: int
    current_batch_index: int
    scores: Dict[str, Deque[Score]]
    train_losses: Deque[Union[float, Dict[str, float]]]
    _primary_objective: str

    @property
    def primary_scores(self) -> List[Score]:
        return self.scores[self._primary_objective]

    @property
    def current_scores(self) -> Dict[str, Score]:
        return {score_name: scores[self.current_epoch - 1] for score_name, scores in self.scores.items() if self.current_epoch > 0}

@dataclass
class MultiTrainingContext(Generic[TInput, TTarget, TModel]):
    current_dataset_index: int

class BatchTrainingPlugin(Generic[TInput, TTarget, TModel], ABC):
    pass

class PreMultiLoop(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_multi_loop(self, logger: Logger, context: MultiTrainingContext):
        pass

class PostMultiLoop(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_multi_loop(self, logger: Logger, context: MultiTrainingContext):
        pass

class PreMultiTrainStep(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_multi_train_step(self, logger: Logger, context: MultiTrainingContext):
        pass

class PostMultiTrainStep(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_multi_train_step(self, logger: Logger, context: MultiTrainingContext):
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
    def pre_epoch(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        pass

class PostEpoch(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_epoch(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        pass

class PreTrain(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_train(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        pass

class PostTrain(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_train(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        pass

