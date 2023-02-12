from logging import Logger
import logging
from typing import Any, Callable, Dict, Optional
from ray import tune
from ray.tune.schedulers import TrialScheduler
from .tuning_service import TuningService
from ray.air.config import RunConfig, ScalingConfig

__all__ = ['RayTuneService']

class RayTuneService(TuningService):
    def __init__(self, resource_config: Dict[str, Any], tune_config: tune.TuneConfig, run_config: RunConfig, logger: Optional[Logger]=None):
        super().__init__()

        if resource_config is None:
            raise TypeError('resource_config')

        if tune_config is None:
            raise TypeError('tune_config')

        if run_config is None:
            raise TypeError('run_config')

        self.__resource_config = resource_config
        self.__tune_config = tune_config
        self.__run_config = run_config
        self.__logger = logger if not logger is None else logging.getLogger()

    async def tune(self, training_function: Callable[[Dict[str, Any]], None], params: Dict[str, Any], logger: Optional[Logger] = None) -> None:
        logger = logger if not logger is None else self.__logger

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(training_function),
                resources=self.__resource_config,
            ),
            tune_config=self.__tune_config,
            param_space=params,
            run_config=self.__run_config
        )

        self.__logger.info('Starting hyperparameter tuning...')

        results = tuner.fit()

        self.__logger.info('Finished hyperparameter tuning.')