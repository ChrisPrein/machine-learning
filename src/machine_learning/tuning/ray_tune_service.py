import asyncio
from functools import partial
from logging import Logger
import logging
from typing import Any, Callable, Dict, Optional, Tuple
from ray import tune
from ray.tune.schedulers import TrialScheduler

from ..modeling.model import TInput, TTarget
from ..training import TrainingService
from .tuning_service import Dataset, TModel, TrainingDataset, TuningService

__all__ = ['RayTuneService']

class RayTuneService(TuningService[TInput, TTarget, TModel]):
    def __init__(self, scheduler: TrialScheduler, resource_config: Dict[str, Any], metric: str, mode: str, num_samples: int, logger: Optional[Logger]=None):
        super().__init__()

        if scheduler is None:
            raise TypeError('scheduler')

        if resource_config is None:
            raise TypeError('resource_config')

        if metric is None:
            raise TypeError('metric')

        if mode is None:
            raise TypeError('mode')

        if num_samples is None:
            raise TypeError('num_samples')

        self.__scheduler = scheduler
        self.__resource_config = resource_config
        self.__metric = metric
        self.__mode = mode
        self.__num_samples = num_samples
        self.__logger = logger if not logger is None else logging.getLogger()

    async def tune(self, training_function: Callable[[Dict[str, Any]], None], params: Dict[str, Any], logger: Optional[Logger] = None) -> None:
        logger = logger if not logger is None else self.__logger

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(training_function),
                resources=self.__resource_config,
            ),
            tune_config=tune.TuneConfig(
                metric=self.__metric,
                mode=self.__mode,
                scheduler=self.__scheduler,
                num_samples=self.__num_samples
            ),
            param_space=params
        )

        self.__logger.info('Starting hyperparameter tuning...')

        results = tuner.fit()

        self.__logger.info('Finished hyperparameter tuning.')