import asyncio
from logging import Logger
from .. import PreTune, TuningContext
from ray import tune
from .repositories import TunerRepository

class TuningCheckpointPlugin(PreTune):
    def __init__(self, tuner_repository: TunerRepository, event_loop: asyncio.AbstractEventLoop = None):
        if tuner_repository is None:
            raise TypeError('trainer_repository')

        self.tuner_repository: TunerRepository = tuner_repository
        self.event_loop: asyncio.AbstractEventLoop = event_loop if event_loop != None else asyncio.get_event_loop()

    def pre_tune(self, logger: Logger, tuning_context: TuningContext):
        logger.info('Loading tuner checkpoint...')

        tuning_context.tuner = self.event_loop.run_until_complete(self.tuner_repository.get(tuning_context.run_config.name))

        logger.info('Tuner checkpoint loaded!')