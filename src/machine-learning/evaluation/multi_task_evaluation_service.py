from abc import ABC, abstractmethod
from argparse import ArgumentError
import inspect
from typing import TypeVar, List, Generic, Optional, Dict
from datetime import timedelta, datetime
from .abstractions.evaluation_context import *
from .abstractions.evaluation_metric import *
from .abstractions.evaluation_service import *
from .default_evaluation_context import *
import asyncio
import asyncio.tasks
import asyncio.futures
from torch.utils.data import DataLoader

class MultiTaskEvaluationService(EvaluationService[DefaultEvaluationContext[TModel]], ABC):
    def __init__(self):
        self.__semaphore = asyncio.Semaphore(1)

    async def evaluate(self, model: TModel, evaluation_data_loader: DataLoader, evaluation_metrics: Dict[str, EvaluationMetric[DefaultEvaluationContext[TModel]]]) -> Dict[str, float]:
        if model is None:
            raise ValueError("model")

        if evaluation_data_loader is None:
            raise ValueError("evaluation_data_loader")

        if evaluation_metrics is None:
            raise ValueError("evaluation_metrics")

        evaluation_context: DefaultEvaluationContext[TModel] = DefaultEvaluationContext[TModel](model)

        for batch_index, batch in enumerate(evaluation_data_loader):
            evaluation_context.predictions.append(model.predict_batch(batch))

        result: EvaluationResult = EvaluationResult[TModel, TMember, TData](dict(), self.config.evaluation_metrics, model, model_result)

        for evaluation_metric in self.config.evaluation_metrics:
            result.evaluation_score[evaluation_metric.name] = evaluation_metric.calculate_score(model_result, data)

        return result