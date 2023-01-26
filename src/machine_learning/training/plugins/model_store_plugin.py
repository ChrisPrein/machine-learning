from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from typing import Dict, Union
from ...evaluation.evaluation_context import TModel
from ...modeling import TInput, TTarget, TOutput
from ..batch_training_service import PostEpoch, TTrainer, TrainingContext
from .repositories import ModelRepository, ModelMetadataRepository, ModelMetadata
import asyncio

BEST_MODEL_NAME: str = "best-model"
METADATA_NAME: str = "best-model-meta"

class ModelStorePlugin(PostEpoch[TInput, TTarget, TOutput, TModel, TTrainer]):
    def __init__(self, model_repository: ModelRepository[TModel], metadata_repository: ModelMetadataRepository, loss_key: str = None, event_loop: asyncio.AbstractEventLoop = None):
        if model_repository is None:
            raise TypeError('model_repository')

        if metadata_repository is None:
            raise TypeError('metadata_repository')

        self.model_repository: ModelRepository[TModel] = model_repository
        self.metadata_repository: ModelMetadataRepository = metadata_repository
        self.loss_key: str = loss_key
        self.event_loop: asyncio.AbstractEventLoop = event_loop if event_loop != None else asyncio.get_event_loop()

        self.best_model: TModel = self.event_loop.run_until_complete(self.model_repository.get(BEST_MODEL_NAME))
        self.metadata: ModelMetadata = self.event_loop.run_until_complete(self.metadata_repository.get(METADATA_NAME))

        if self.metadata is None:
            self.metadata = ModelMetadata(float('inf'))

    def post_epoch(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        current_loss: Union[float, Dict[str, float]] = training_context.train_losses[-1]

        if isinstance(current_loss, dict):
            current_loss = current_loss[self.loss_key] if self.loss_key != None else current_loss.items()[0]

        if current_loss < self.metadata.loss:
            logger.info('Saving current best model...')

            self.best_model = training_context.model
            self.metadata.loss = current_loss

            self.event_loop.create_task(self.model_repository.save(self.best_model, BEST_MODEL_NAME))
            self.event_loop.create_task(self.metadata_repository.save(self.metadata, METADATA_NAME))

            logger.info('Current best model saved!')