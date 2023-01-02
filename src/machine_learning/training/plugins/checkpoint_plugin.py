from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
import pathlib
import pickle
from typing import Generic
from ...training import TModel, TTrainer, PreLoop, PostEpoch, TrainingContext
from ...modeling import TInput, TTarget
from ...repositories import ModelRepository, TrainingCheckpointRepository, TrainerRepository
import asyncio

@dataclass
class TrainingCheckpoint:
    current_epoch: int
    current_batch_index: int
    continue_training: bool

LATEST_MODEL_NAME: str = "latest-model"
LATEST_TRAINER_NAME: str = "latest-trainer"
LATEST_TRAINING_CHECKPOINT_NAME: str = "latest-checkpoint"

class CheckpointPlugin(PostEpoch[TInput, TTarget, TModel, TTrainer], PreLoop[TInput, TTarget, TModel, TTrainer]):
    def __init__(self, model_checkpoint_repository: ModelRepository, training_checkpoint_repository: TrainingCheckpointRepository, trainer_repository: TrainerRepository[TTrainer], event_loop: asyncio.AbstractEventLoop = None):
        if model_checkpoint_repository is None:
            raise TypeError('model_checkpoint_repository')

        if training_checkpoint_repository is None:
            raise TypeError('training_checkpoint_repository')

        if trainer_repository is None:
            raise TypeError('trainer_repository')

        self.model_checkpoint_repository: ModelRepository = model_checkpoint_repository
        self.training_checkpoint_repository: TrainingCheckpointRepository = training_checkpoint_repository
        self.trainer_repository: TrainerRepository = trainer_repository
        self.event_loop: asyncio.AbstractEventLoop = event_loop if event_loop != None else asyncio.get_event_loop()
        
        self.checkpoint: TrainingCheckpoint = self.event_loop.run_until_complete(self.training_checkpoint_repository.get(LATEST_TRAINING_CHECKPOINT_NAME))

        if self.checkpoint is None:
            self.checkpoint = TrainingCheckpoint(current_epoch=0, current_batch_index=0, continue_training=True)

    def post_epoch(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TModel, TTrainer]):
        logger.info('Creating checkpoint...')

        self.checkpoint.current_epoch = training_context.current_epoch
        self.checkpoint.current_batch_index = training_context.current_batch_index
        self.checkpoint.continue_training = training_context.continue_training

        self.event_loop.create_task(self.training_checkpoint_repository.save(self.checkpoint, LATEST_TRAINING_CHECKPOINT_NAME))
        self.event_loop.create_task(self.model_checkpoint_repository.save(training_context.model, LATEST_MODEL_NAME))

        logger.info('Checkpoint created!')

    def pre_loop(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TModel, TTrainer]):
        logger.info('Loading checkpoint...')

        training_context.current_epoch = self.checkpoint.current_epoch
        training_context.current_batch_index = self.checkpoint.current_batch_index
        training_context.continue_training = self.checkpoint.continue_training

        latest_model = self.event_loop.run_until_complete(self.model_checkpoint_repository.get(LATEST_MODEL_NAME))
        latest_trainer = self.event_loop.run_until_complete(self.trainer_repository.get(LATEST_TRAINER_NAME))
        
        if latest_model != None:
            training_context.model = latest_model

        if latest_trainer != None:
            training_context.trainer = latest_trainer

        logger.info('Checkpoint loaded!')