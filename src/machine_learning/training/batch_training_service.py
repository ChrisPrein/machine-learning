from abc import ABC, abstractmethod
from argparse import ArgumentError
from dataclasses import dataclass
import enum
import inspect
from typing import TypeVar, List, Generic, Optional, Dict, Tuple
from datetime import timedelta, datetime
from unittest.mock import patch

from numpy import number

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
    def __init__(self, event_loop: Optional[asyncio.AbstractEventLoop] = None):
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()

    async def train(self, model: TModel, dataset: Dataset[Tuple[TInput, TTarget]], stop_conditions: Dict[str, StopCondition[TrainingContext[TModel]]]) -> TModel:
        if model is None:
            raise ValueError("model")

        if dataset is None:
            raise ValueError("dataset")

        if stop_conditions is None:
            raise ValueError("stop_conditions")

        evaluation_context: DefaultEvaluationContext[TInput, TTarget, TModel] = DefaultEvaluationContext[TInput, TTarget, TModel](model)

        prediction_futures: List[asyncio.Future] = [self.__event_loop.run_in_executor(None, lambda: model.predict_batch(input_batch=[sample[0] for sample in batch])) for batch in evaluation_data_loader]
        completed, pending = self.__event_loop.run_until_complete(asyncio.wait(prediction_futures))

        for t in completed:
            evaluation_context.predictions.extend(t.result()) 

        result: Dict[str, float] = {}

        for i, (name, evaluation_metric) in enumerate(evaluation_metrics.items()):
            result[name] = evaluation_metric.calculate_score(context=evaluation_context)

        return result