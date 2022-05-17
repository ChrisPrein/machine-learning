from logging import Logger
import logging
from typing import Any, Callable, Coroutine, TypeVar, List, Generic, Optional, Dict, Tuple, Union
from uuid import UUID
import uuid

from attr import asdict
from ..modeling.abstractions.model import Model, TInput, TTarget
from .abstractions.evaluation_metric import EvaluationContext, EvaluationMetric, Prediction, TModel
from .abstractions.evaluation_service import EvaluationService
from .default_evaluation import default_evaluation
import asyncio
import asyncio.tasks
import asyncio.futures
from dataset_handling.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import nest_asyncio

EVALUATION_LOGGER_NAME = "evaluation"

nest_asyncio.apply()

class MultiTaskEvaluationService(EvaluationService[TInput, TTarget, TModel]):
    def __init__(self, evaluation_hook: Callable[[Logger, TModel, List[TInput], List[TTarget]], List[TTarget]] = default_evaluation, logger: Optional[Logger]=None, batch_size: int = 1, drop_last: bool = True, event_loop: Optional[asyncio.AbstractEventLoop] = None):
        if logger is None:
            self.__logger: Logger = logging.getLogger()
        else:
            self.__logger: Logger = logger.getChild(EVALUATION_LOGGER_NAME)

        if evaluation_hook is None:
            raise ValueError("evaluation_hook")
        
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()
        self.__batch_size: int = batch_size
        self.__drop_last: bool = drop_last
        self.__evaluation_hook: Callable[[Logger, TModel, List[TInput], List[TTarget]], List[TTarget]] = evaluation_hook

    def __predict_batch(self, model: TModel, batch: List[Tuple[TInput, TTarget]]) -> List[Prediction]:
        inputs: List[TInput] = [sample[0] for sample in batch]
        targets: List[TInput] = [sample[1] for sample in batch]
        predictions: List[TTarget] = self.__evaluation_hook(self.__logger, model, inputs, targets)

        combined: List[Tuple[TInput, TTarget, TTarget]] = zip(inputs, predictions, targets)

        return [Prediction(result[0], result[1], result[2]) for result in combined]

    async def evaluate(self, model: TModel, evaluation_dataset: Union[Tuple[str, Dataset[Tuple[TInput, TTarget]]], Dataset[Tuple[TInput, TTarget]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget, TModel]]) -> Dict[str, float]:
        if model is None:
            raise ValueError("model")

        if evaluation_dataset is None:
            raise ValueError("evaluation_dataset")

        if evaluation_metrics is None:
            raise ValueError("evaluation_metrics")

        evaluation_context: EvaluationContext[TInput, TTarget, TModel] = EvaluationContext[TInput, TTarget, TModel](model, [])

        dataset: Dataset[Tuple[TInput, TTarget]] = None

        if isinstance(evaluation_dataset, Tuple):
            dataset = evaluation_dataset[1]
        else:
            dataset = evaluation_dataset

        data_loader: DataLoader[Tuple[TInput, TTarget]] = DataLoader[Tuple[TInput, TTarget]](dataset=dataset, batch_size=self.__batch_size, drop_last=self.__drop_last)

        self.__logger.info('Starting evaluation loop...')

        prediction_futures: List[asyncio.Future] = [self.__event_loop.run_in_executor(None, lambda: self.__predict_batch(model, batch)) for batch in data_loader]
        predictions: List[List[Prediction]] = await asyncio.gather(*prediction_futures, loop=self.__event_loop)

        for t in predictions:
            evaluation_context.predictions.extend(t) 

        result: Dict[str, float] = {}

        for i, (name, evaluation_metric) in enumerate(evaluation_metrics.items()):
            result[name] = evaluation_metric.calculate_score(context=evaluation_context)

        self.__logger.info('Finished evaluation loop.')
        self.__logger.info(evaluation_context)

        return result

    async def __evaluate(self, model: TModel, evaluation_dataset: Tuple[str, Dataset[Tuple[TInput, TTarget]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget, TModel]]) -> Tuple[str, Dict[str, float]]:
        result = await self.evaluate(model, evaluation_dataset, evaluation_metrics)

        return (evaluation_dataset[0], result)

    async def evaluate_on_multiple_datasets(self, model: TModel, evaluation_datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget, TModel]]) -> Dict[str, Dict[str, float]]:
        self.__logger.info(f"starting evaluation on {len(evaluation_datasets)} datasets...")
        
        experiment_tasks: List[Coroutine[Any, Any, Tuple[str, Dict[str, float]]]] = [self.__evaluate(model, dataset, evaluation_metrics, self.__logger) for dataset in evaluation_datasets.items()]

        experiment_results: List[Tuple[str, Dict[str, float]]] = await asyncio.gather(*experiment_tasks, loop=self.__event_loop)

        results = dict(experiment_results)

        self.__logger.info(f"finished evaluation on {len(evaluation_datasets)} datasets.")

        return results