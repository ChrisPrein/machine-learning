from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from typing import Any, Dict, Optional, TypeGuard, Union

from ...evaluation.evaluation_context import TModel
from ...modeling import TInput, TTarget, TOutput
from ..batch_training_service import PostEpoch, TTrainer, TrainingContext
from .repositories import ModelRepository, ModelMetadataRepository, ModelMetadata
from .validation_plugin import PostValidationPlugin
from ...evaluation import EvaluationResult
import asyncio

BEST_MODEL_NAME: str = "best-model"
METADATA_NAME: str = "best-model-meta"

def is_nested_dict(val: Dict[str, Any]) -> TypeGuard[Dict[str, Dict[str, float]]]:
    return all(isinstance(value, dict) for value in val.values())

class ModelStorePlugin(PostValidationPlugin[TInput, TTarget, TOutput, TModel, TTrainer]):
    def __init__(self, model_repository: ModelRepository[TModel], metadata_repository: ModelMetadataRepository, metric_key: str, dataset_name: Optional[str] = None, event_loop: asyncio.AbstractEventLoop = None):
        if model_repository is None:
            raise TypeError('model_repository')

        if metadata_repository is None:
            raise TypeError('metadata_repository')

        self.model_repository: ModelRepository[TModel] = model_repository
        self.metadata_repository: ModelMetadataRepository = metadata_repository
        self.metric_key: str = metric_key
        self.dataset_name: Optional[str] = dataset_name
        self.event_loop: asyncio.AbstractEventLoop = event_loop if event_loop != None else asyncio.get_event_loop()

        self.best_model: TModel = self.event_loop.run_until_complete(self.model_repository.get(BEST_MODEL_NAME))
        self.metadata: ModelMetadata = self.event_loop.run_until_complete(self.metadata_repository.get(METADATA_NAME))

        if self.metadata is None:
            self.metadata = ModelMetadata(None, None)

    def post_validation(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer], validation_result: EvaluationResult):
        is_nested_result: bool = is_nested_dict(validation_result)

        if is_nested_result and self.dataset_name is None:
            raise TypeError('Datasetname has to be set for nested validation results.')

        current_performance: float = validation_result[self.metric_key] if not is_nested_result else validation_result[self.dataset_name][self.metric_key]

        if self.metadata.performance is None:
            best_performance: float = float("-inf")
        else:
            best_performance: float = self.metadata.performance[self.metric_key] if not is_nested_result else self.metadata.performance[self.dataset_name][self.metric_key]

        if current_performance > best_performance:
            logger.info('Saving current best model...')

            self.best_model = training_context.model
            self.metadata.loss = training_context.train_losses[-1]
            self.metadata.performance = validation_result

            self.event_loop.create_task(self.model_repository.save(self.best_model, BEST_MODEL_NAME))
            self.event_loop.create_task(self.metadata_repository.save(self.metadata, METADATA_NAME))

            logger.info('Current best model saved!')