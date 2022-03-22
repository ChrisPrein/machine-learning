from abc import ABC, abstractmethod
from argparse import ArgumentError
from dataclasses import dataclass
import enum
import inspect
from logging import Logger
from typing import TypeVar, List, Generic, Optional, Dict, Tuple
from datetime import timedelta, datetime
from unittest.mock import patch

from numpy import number

from .default_training_context import DefaultTrainingContext

from .abstractions.stop_condition import StopCondition

from .abstractions.training_context import TModel, TrainingContext
from ..modeling.abstractions.model import TInput, TTarget
from .abstractions.training_service import TrainingService
import asyncio
import asyncio.tasks
import asyncio.futures
from dataset_handling.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

import nest_asyncio
nest_asyncio.apply()

class BatchTrainingService(TrainingService[TInput, TTarget, TModel, TrainingContext[TModel]], ABC):
    def __init__(self, logger: Logger, batch_size: Optional[int], drop_last: bool = True, event_loop: Optional[asyncio.AbstractEventLoop] = None, max_epochs: int = 100, max_iterations: int = 10000):
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()
        self.__max_epochs: int = max_epochs
        self.__max_iterations: int = max_iterations
        self.__batch_size: Optional[int] = batch_size
        self.__drop_last: bool = drop_last
        self.__logger: Logger = Logger

    def is_any_stop_condition_satisfied(self, training_context: DefaultTrainingContext[TModel], stop_conditions: Dict[str, StopCondition[TrainingContext[TModel]]]) -> bool:
        self.__logger.info("Checking stop conditions.")

        is_any_satisfied: bool = False

        if training_context.current_epoch > self.__max_epochs: 
            self.__logger.info("Max number of epochs reached.")
            is_any_satisfied = True
        elif training_context.current_iteration > self.__max_iterations:
            self.__logger.info("Max number of iterations reached.")
        else:
            for key, condition in stop_conditions.items():
                is_any_satisfied |= condition.is_satisfied(training_context)

                if(is_any_satisfied):
                    self.__logger('Condition named "{key}" is satisfied'.format(key=key))

                break

        self.__logger.info("Done checking stop conditions.")

        return is_any_satisfied

    async def train(self, model: TModel, dataset: Dataset[Tuple[TInput, TTarget]], stop_conditions: Dict[str, StopCondition[TrainingContext[TModel]]]) -> TModel:
        if model is None:
            raise ValueError("model")

        if dataset is None:
            raise ValueError("dataset")

        if stop_conditions is None:
            raise ValueError("stop_conditions")

        batch_size: int = len(dataset) if self.__batch_size is None else self.__batch_size

        evaluation_context: DefaultTrainingContext[TModel] = DefaultTrainingContext[TModel](model)
        data_loader: DataLoader[Tuple[TInput, TTarget]] = DataLoader[Tuple[TInput, TTarget]](dataset=dataset, batch_size=batch_size, drop_last=self.__drop_last)

        self.__logger.info("Starting training loop.")

        while not self.is_any_stop_condition_satisfied(training_context=evaluation_context, stop_conditions=stop_conditions):
            for batch_index, batch in enumerate(data_loader):
                inputs: List[TInput] = [value[0] for value in batch]
                targets: List[TTarget] = [value[1] for value in batch]
                model.train_batch(input_batch=inputs, target_batch=targets)

        self.__logger.info("Finished training loop.")

        return model