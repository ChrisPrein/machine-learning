from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
import pathlib
import pickle
from typing import Generic
from machine_learning.training import *
from machine_learning import TInput, TTarget

@dataclass
class TrainingCheckpoint:
    current_epoch: int
    current_batch_index: int
    continue_training: bool

class ModelCheckpointRepository(Generic[TModel], ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get(self, name: str) -> TModel: ...

    @abstractmethod
    def save(self, model: TModel, name: str): ...

class TrainingCheckpointRepository(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get(self, name: str) -> TrainingCheckpoint: ...

    @abstractmethod
    def save(self, checkpoint: TrainingCheckpoint, name: str): ...

LATEST_MODEL_NAME: str = "latest-model"
LATEST_TRAINING_CHECKPOINT_NAME: str = "latest-checkpoint"

class CheckpointPlugin(PostEpoch[TInput, TTarget, TModel], PreLoop[TInput, TTarget, TModel]):
    def __init__(self, model_checkpoint_repository: ModelCheckpointRepository, training_checkpoint_repository: TrainingCheckpointRepository):
        if model_checkpoint_repository is None:
            raise TypeError('model_checkpoint_repository')

        if training_checkpoint_repository is None:
            raise TypeError('training_checkpoint_repository')

        self.model_checkpoint_repository: ModelCheckpointRepository = model_checkpoint_repository
        self.training_checkpoint_repository: TrainingCheckpointRepository = training_checkpoint_repository
        
        self.checkpoint: TrainingCheckpoint = self.training_checkpoint_repository.get(LATEST_TRAINING_CHECKPOINT_NAME)

        if self.checkpoint is None:
            self.checkpoint = TrainingCheckpoint(current_epoch=0, current_batch_index=0, continue_training=True)

    def post_epoch(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        self.checkpoint.current_epoch = trainingContext.current_epoch
        self.checkpoint.current_batch_index = trainingContext.current_batch_index
        self.checkpoint.continue_training = trainingContext.continue_training

        self.training_checkpoint_repository.save(self.checkpoint, LATEST_TRAINING_CHECKPOINT_NAME)
        self.model_checkpoint_repository.save(trainingContext.model, LATEST_MODEL_NAME)

    def pre_loop(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        trainingContext.current_epoch = self.checkpoint.current_epoch
        trainingContext.current_batch_index = self.checkpoint.current_batch_index
        trainingContext.continue_training = self.checkpoint.continue_training

        latest_model = self.model_checkpoint_repository.get(LATEST_MODEL_NAME)
        
        if latest_model != None:
            trainingContext.model = latest_model