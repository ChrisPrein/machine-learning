from logging import Logger

from ...evaluation import EvaluationResult
from ...modeling import TInput, TTarget
from .validation_plugin import PostValidationPlugin
from ...training.batch_training_service import TTrainer, TrainingContext, TModel
from typing import *
from abc import ABC, abstractmethod

__all__ = ['StopCondition', 'StopConditions', 'EarlyStoppingPlugin']

class StopCondition(Generic[TInput, TTarget, TModel, TTrainer], ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def is_satisfied(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TModel, TTrainer], validation_result: EvaluationResult) -> bool:
        pass

StopConditions = Dict[str, StopCondition[TInput, TTarget, TModel, TTrainer]]

class EarlyStoppingPlugin(PostValidationPlugin[TInput, TTarget, TModel, TTrainer]):
    def __init__(self, stop_conditions: StopConditions):
        if stop_conditions == None:
            raise TypeError('stop_conditions')

        self.stop_conditions: StopConditions = stop_conditions

    def post_validation(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TModel, TTrainer], validation_result: EvaluationResult):
        logger.info("Checking stop conditions...")
        is_any_satisfied: bool = False

        for key, condition in self.stop_conditions.items():
            is_any_satisfied |= condition.is_satisfied(logger, training_context, validation_result)

            if(is_any_satisfied):
                logger.info('Condition named "{key}" is satisfied'.format(key=key))
                training_context.continue_training = False

        logger.info("Finished checking stop conditions.")