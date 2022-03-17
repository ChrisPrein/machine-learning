from abc import ABC, abstractmethod
from argparse import ArgumentError
from dataclasses import dataclass
import enum
import inspect
from typing import TypeVar, List, Generic, Optional, Dict
from datetime import timedelta, datetime

from numpy import number
from .abstractions.evaluation_context import *
from .abstractions.evaluation_metric import *
from .abstractions.evaluation_service import *
from .default_evaluation_context import *
import asyncio
import asyncio.tasks
import asyncio.futures
from torch.utils.data import DataLoader

@dataclass
class BatchPrediction(Generic[TTarget]):
    batch_index: number
    prediction: List[TTarget]

class MultiTaskEvaluationService(EvaluationService[TInput, TTarget, TModel, DefaultEvaluationContext[TTarget, TModel]], ABC):
    def __init__(self, event_loop: Optional[asyncio.AbstractEventLoop] = None):
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()

    async def evaluate(self, model: TModel, evaluation_data_loader: DataLoader[(TInput, TTarget)], evaluation_metrics: Dict[str, EvaluationMetric[DefaultEvaluationContext[TModel]]]) -> Dict[str, float]:
        if model is None:
            raise ValueError("model")

        if evaluation_data_loader is None:
            raise ValueError("evaluation_data_loader")

        if evaluation_metrics is None:
            raise ValueError("evaluation_metrics")

        evaluation_context: DefaultEvaluationContext[TModel] = DefaultEvaluationContext[TModel](model)

        prediction_futures: List[asyncio.Future] = [self.__event_loop.run_in_executor(executor=None, func=model.predict_batch, args=batch) for batch in evaluation_data_loader]
        completed, pending = asyncio.wait(prediction_futures)

        for t in completed:
            evaluation_context.predictions.append(t.result()) 

        result: Dict[str, float] = {}

        for i, (name, evaluation_metric) in enumerate(evaluation_metrics.items()):
            result[name] = evaluation_metric.calculate_score(evaluation_context)

        return result