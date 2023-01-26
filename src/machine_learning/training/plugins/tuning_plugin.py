from logging import Logger

from ...evaluation import EvaluationResult
from ...modeling import TInput, TTarget, TOutput
from .validation_plugin import PostValidationPlugin
from ...training.batch_training_service import TTrainer, TrainingContext, TModel
from typing import *
from ray import tune

__all__ = ['TuningPlugin']

class TuningPlugin(PostValidationPlugin[TInput, TTarget, TOutput, TModel, TTrainer]):
    def __init__(self):
        super().__init__()

    def post_validation(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer], validation_result: EvaluationResult):
        logger.info("Reporting to tuner...")

        tune.report(validation_result)

        logger.info("Reported to tuner.")