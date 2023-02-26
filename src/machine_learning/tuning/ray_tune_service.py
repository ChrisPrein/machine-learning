from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
import logging
from typing import Any, Callable, Dict, Generic, Optional
from ray import tune
from ray.tune.schedulers import TrialScheduler
from .tuning_service import TuningService
from ray.air.config import RunConfig, ScalingConfig
from ray.tune.result_grid import ResultGrid

__all__ = ['RayTuneService', 'TuningContext', 'RayTunePlugin', 'PreTune', 'PostTune']

@dataclass
class TuningContext:
    tuner: tune.Tuner
    run_config: RunConfig

class RayTunePlugin(ABC):
    pass

class PreTune(RayTunePlugin):
    @abstractmethod
    def pre_tune(self, logger: Logger, tuning_context: TuningContext):
        pass

class PostTune(RayTunePlugin):
    @abstractmethod
    def post_tune(self, logger: Logger, tuning_context: TuningContext):
        pass

class RayTuneService(TuningService):
    def __init__(self, resource_config: Dict[str, Any], tune_config: tune.TuneConfig, run_config: RunConfig, logger: Optional[Logger]=None, plugins: Dict[str, RayTunePlugin] = {}, **kwargs):
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

        self.__pre_tune_plugins: Dict[str, PreTune] = dict(filter(lambda plugin: isinstance(plugin[1], PreTune), plugins.items()))
        self.__post_tune_plugins: Dict[str, PostTune] = dict(filter(lambda plugin: isinstance(plugin[1], PostTune), plugins.items()))

    def __execute_pre_tune_plugins(self, logger: Logger, context: TuningContext):
        logger.debug("Executing pre tune plugins...")
        for name, plugin in self.__pre_tune_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.pre_tune(logger, context)

    def __execute_post_tune_plugins(self, logger: Logger, context: TuningContext):
        logger.debug("Executing post tune plugins...")
        for name, plugin in self.__post_tune_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.post_tune(logger, context)

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

        tuning_context: TuningContext = TuningContext(tuner, self.__run_config)

        self.__execute_pre_tune_plugins(logger, tuning_context)

        self.__logger.info('Starting hyperparameter tuning...')

        results: ResultGrid = tuning_context.tuner.fit()

        self.__logger.info('Finished hyperparameter tuning.')

        self.__execute_post_tune_plugins(logger, tuning_context)