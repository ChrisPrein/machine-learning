from logging import Logger

from ...evaluation import EvaluationResult
from ...modeling import TInput, TTarget
from .validation_plugin import PostValidationPlugin
from ...training.batch_training_service import TTrainer, TrainingContext, TModel
from typing import *
from ray import tune

__all__ = ['TuningPlugin']

class TuningPlugin(PostValidationPlugin[TInput, TTarget, TModel, TTrainer]):
    def __init__(self):
        super().__init__()

    def post_validation(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TModel, TTrainer], validation_result: EvaluationResult):
        logger.info("Reporting to tuner...")

        tune.report(validation_result)

        logger.info("Reported to tuner.")