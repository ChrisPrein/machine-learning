from abc import ABC, abstractmethod
import asyncio
from functools import partial
from logging import Logger
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, Generic, Tuple, Union, overload
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, TrialScheduler

from ..evaluation.multi_evaluation_context import Score
from ..modeling.model import TInput, TTarget, Model
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

    async def tune(self, 
        model_factory: Callable[[], TModel], 
        training_service_factory: Callable[[], TrainingService[TInput, TTarget, TModel]], 
        train_dataset: TrainingDataset[TInput, TTarget], 
        params: Dict[str, Any], 
        logger: Optional[Logger] = None) -> Tuple[TModel, Dict[str, Dict[str, Score]]]:

        if isinstance(train_dataset, tuple):
            return await self.__tune(model_factory=model_factory, training_service_factory=training_service_factory, train_dataset=train_dataset, params=params, logger=logger)
        else:
            return await self.__tune(model_factory=model_factory, training_service_factory=training_service_factory, train_dataset=('dataset', train_dataset), params=params, logger=logger)

    async def __tune(self, 
        model_factory: Callable[[], TModel], 
        training_service_factory: Callable[[], TrainingService[TInput, TTarget, TModel]], 
        train_dataset: Tuple[str, Dataset[TInput, TTarget]], 
        params: Dict[str, Any], 
        logger: Optional[Logger] = None) -> Tuple[TModel, Dict[str, Dict[str, Score]]]:

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(partial(self.__train(model_factory=model_factory, training_service_factory=training_service_factory, train_dataset=train_dataset, logger=logger))),
                resources=self.__resource_config
            ),
            tune_config=tune.TuneConfig(
                metric=self.__metric,
                mode=self.__mode,
                scheduler=self.__scheduler,
                num_samples=self.__num_samples
            )
        )

        results = tuner.fit()

        return results

    def __train(self, 
        config, 
        model_factory: Callable[[], TModel], 
        training_service_factory: Callable[[], TrainingService[TInput, TTarget, TModel]], 
        train_dataset: Tuple[str, Dataset[TInput, TTarget]], 
        logger: Optional[Logger] = None):

        logger = logger if not logger is None else self.__logger

        model: TModel = model_factory(**config["model"])
        training_service: TrainingService[TInput, TTarget, TModel] = training_service_factory(**config["training_service"])

        event_loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()

        event_loop.run_until_complete(training_service.train(model, train_dataset, logger))