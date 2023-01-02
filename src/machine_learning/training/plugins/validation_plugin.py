from abc import abstractmethod
import asyncio
from logging import Logger
from typing import *
from ...evaluation import *
from ...modeling import *
from ..batch_training_service import TTrainer, TrainingContext, PostEpoch

__all__ = ['PostValidationPlugin', 'PreValidationPlugin', 'ValidationPlugins', 'ValidationPlugin']

class PostValidationPlugin(Generic[TInput, TTarget, TModel, TTrainer]):
    @abstractmethod
    def post_validation(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TModel, TTrainer], validation_result: EvaluationResult):
        pass

class PreValidationPlugin(Generic[TInput, TTarget, TModel, TTrainer]):
    @abstractmethod
    def pre_validation(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TModel, TTrainer]):
        pass

ValidationPlugins = Union[PreValidationPlugin, PostValidationPlugin]

class ValidationPlugin(PostEpoch[TInput, TTarget, TModel, TTrainer]):
    def __init__(self, evaluation_service: EvaluationService[TInput, TTarget, TModel], validation_datasets: EvaluationDataset, validation_metrics: EvaluationMetrics, event_loop: asyncio.AbstractEventLoop = None, plugins: Dict[str, ValidationPlugins] = {}):
        if evaluation_service == None:
            raise ValueError('evaluaiton_service')

        if validation_datasets == None:
            raise ValueError('validation_datasets')

        self.__evaluation_service: EvaluationService[TInput, TTarget, TModel] = evaluation_service
        self.__validation_datasets: EvaluationDataset = validation_datasets
        self.__validation_metrics: EvaluationMetrics = validation_metrics
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if event_loop != None else asyncio.get_event_loop()

        self.__pre_validation_plugins: Dict[str, PreValidationPlugin[TInput, TTarget, TModel, TTrainer]] = dict(filter(lambda plugin: isinstance(plugin[1], PreValidationPlugin), plugins.items()))
        self.__post_validation_plugins: Dict[str, PostValidationPlugin[TInput, TTarget, TModel, TTrainer]] = dict(filter(lambda plugin: isinstance(plugin[1], PostValidationPlugin), plugins.items()))

    def __execute_pre_validation_plugins(self, logger: Logger, context: TrainingContext[TInput, TTarget, TModel, TTrainer]):
        logger.debug("Executing pre validation plugins...")
        for name, plugin in self.__pre_validation_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.pre_validation(logger, context)

    def __execute_post_validation_plugins(self, logger: Logger, context: TrainingContext[TInput, TTarget, TModel, TTrainer], validation_result: EvaluationResult):
        logger.debug("Executing post validation plugins...")
        for name, plugin in self.__post_validation_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.post_validation(logger, context, validation_result)

    def post_epoch(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel, TTrainer]):
        logger.info("Validating current model.")
        self.__execute_pre_validation_plugins(logger, trainingContext)
        validation_result: EvaluationResult = self.__event_loop.run_until_complete(self.__evaluation_service.evaluate(model=trainingContext.model, evaluation_dataset=self.__validation_datasets, evaluation_metrics=self.__validation_metrics, logger=logger))
        self.__execute_post_validation_plugins(logger, trainingContext, validation_result)
        logger.info("finished validation of current model.")

