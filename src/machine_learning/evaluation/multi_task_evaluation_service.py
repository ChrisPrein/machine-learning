from abc import ABC, abstractmethod
from argparse import ArgumentError
from dataclasses import dataclass
import enum
import inspect
from typing import TypeVar, List, Generic, Optional, Dict, Tuple
from datetime import timedelta, datetime
from unittest.mock import patch

from numpy import number
from ..modeling.abstractions.model import TInput, TTarget
from .abstractions.evaluation_metric import EvaluationMetric
from .abstractions.evaluation_service import EvaluationService
from .abstractions.evaluation_context import TModel
from .default_evaluation_context import DefaultEvaluationContext
import asyncio
import asyncio.tasks
import asyncio.futures
from dataset_handling.dataloader import DataLoader

import nest_asyncio
nest_asyncio.apply()

class MultiTaskEvaluationService(EvaluationService[TInput, TTarget, TModel, DefaultEvaluationContext[TInput, TTarget, TModel]], ABC):
    def __init__(self, event_loop: Optional[asyncio.AbstractEventLoop] = None):
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()

    async def evaluate(self, model: TModel, evaluation_data_loader: DataLoader[Tuple[TInput, TTarget]], evaluation_metrics: Dict[str, EvaluationMetric[DefaultEvaluationContext[TInput, TTarget, TModel]]]) -> Dict[str, float]:
        if model is None:
            raise ValueError("model")

        if evaluation_data_loader is None:
            raise ValueError("evaluation_data_loader")

        if evaluation_metrics is None:
            raise ValueError("evaluation_metrics")

        evaluation_context: DefaultEvaluationContext[TInput, TTarget, TModel] = DefaultEvaluationContext[TInput, TTarget, TModel](model)

        prediction_futures: List[asyncio.Future] = [self.__event_loop.run_in_executor(None, lambda: model.predict_batch(input_batch=[sample[0] for sample in batch])) for batch in evaluation_data_loader]
        completed, pending = self.__event_loop.run_until_complete(asyncio.wait(prediction_futures))

        for t in completed:
            evaluation_context.predictions.extend(t.result()) 

        result: Dict[str, float] = {}

        for i, (name, evaluation_metric) in enumerate(evaluation_metrics.items()):
            result[name] = evaluation_metric.calculate_score(context=evaluation_context)

        return result