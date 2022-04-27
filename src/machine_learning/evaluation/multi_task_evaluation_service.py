from logging import Logger
import logging
from typing import Any, Coroutine, TypeVar, List, Generic, Optional, Dict, Tuple
from uuid import UUID
import uuid
from ..modeling.abstractions.model import Model, TInput, TTarget
from .abstractions.evaluation_metric import EvaluationContext, EvaluationMetric, Prediction, TModel
from .abstractions.evaluation_service import EvaluationService
import asyncio
import asyncio.tasks
import asyncio.futures
from dataset_handling.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import nest_asyncio

STARTING_EVALUATION = 80
FINISHED_EVALUATION = 81
STARTING_MULTI_DATASET_EVALUATION = 82
FINISHED_MULTI_DATASET_EVALUATION = 83

EVALUATION_LOGGER_NAME = "evaluation"

nest_asyncio.apply()

class MultiTaskEvaluationService(EvaluationService[TInput, TTarget, TModel]):
    def __init__(self, logger: Optional[Logger]=None, batch_size: Optional[int] = None, drop_last: bool = True, event_loop: Optional[asyncio.AbstractEventLoop] = None):
        if logger is None:
            self.__logger: Logger = logging.getLogger()
        else:
            self.__logger: Logger = logger.getChild(EVALUATION_LOGGER_NAME)
        
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()
        self.__batch_size: Optional[int] = batch_size
        self.__drop_last: bool = drop_last

    def __predict_batch(self, model: TModel, batch: List[Tuple[TInput, TTarget]]) -> List[Prediction]:
        inputs: List[TInput] = [sample[0] for sample in batch]
        targets: List[TInput] = [sample[1] for sample in batch]
        predictions: List[TTarget] = model.predict_batch(input_batch=inputs)

        combined: List[Tuple[TInput, TTarget, TTarget]] = zip(inputs, predictions, targets)

        return [Prediction(result[0], result[1], result[2]) for result in combined]

    async def evaluate(self, model: TModel, evaluation_dataset: Dataset[Tuple[TInput, TTarget]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget, TModel]], logger: Optional[Logger] = None) -> Dict[str, float]:
        evaluation_run_id: UUID = uuid.uuid4()

        if logger is None:
            evaluation_run_logger: Logger = self.__logger.getChild(str(evaluation_run_id))
        else:
            evaluation_run_logger: Logger = logger.getChild(str(evaluation_run_id))
        
        if model is None:
            raise ValueError("model")

        if evaluation_dataset is None:
            raise ValueError("evaluation_dataset")

        if evaluation_metrics is None:
            raise ValueError("evaluation_metrics")

        batch_size: int = len(evaluation_dataset) if self.__batch_size is None else self.__batch_size

        evaluation_context: EvaluationContext[TInput, TTarget, TModel] = EvaluationContext[TInput, TTarget, TModel](model, [])
        data_loader: DataLoader[Tuple[TInput, TTarget]] = DataLoader[Tuple[TInput, TTarget]](dataset=evaluation_dataset, batch_size=batch_size, drop_last=self.__drop_last)

        evaluation_run_logger.log(STARTING_EVALUATION, {"model": model, "evaluation_dataset": evaluation_dataset, "evaluation_metrics": evaluation_metrics, "batch_size": batch_size})
        evaluation_run_logger.info('Starting evaluation loop...')

        prediction_futures: List[asyncio.Future] = [self.__event_loop.run_in_executor(None, lambda: self.__predict_batch(model, batch)) for batch in data_loader]
        completed, pending = self.__event_loop.run_until_complete(asyncio.wait(prediction_futures))

        for t in completed:
            evaluation_context.predictions.extend(t.result()) 

        result: Dict[str, float] = {}

        for i, (name, evaluation_metric) in enumerate(evaluation_metrics.items()):
            result[name] = evaluation_metric.calculate_score(context=evaluation_context)

        evaluation_run_logger.log(FINISHED_EVALUATION, {"model": model, "evaluation_dataset": evaluation_dataset, "evaluation_metrics": evaluation_metrics, "result": result, "batch_size": batch_size})
        evaluation_run_logger.info('Finished evaluation loop.')

        return result

    async def __evaluate(self, model: TModel, evaluation_dataset: Tuple[str, Dataset[Tuple[TInput, TTarget]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget, TModel]], logger: Optional[Logger] = None) -> Tuple[str, Dict[str, float]]:
        result = await self.evaluate(model, evaluation_dataset[1], evaluation_metrics, logger)

        return (evaluation_dataset[0], result)

    async def evaluate_on_multiple_datasets(self, model: TModel, evaluation_datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget, TModel]]) -> Dict[str, Dict[str, float]]:
        multi_evaluation_run_id: UUID = uuid.uuid4()

        multi_evaluation_run_logger: Logger = self.__logger.getChild(str(multi_evaluation_run_id))

        multi_evaluation_run_logger.log(STARTING_MULTI_DATASET_EVALUATION, {"model": model, "evaluation_datasets": evaluation_datasets, "evaluation_metrics": evaluation_metrics, "batch_size": self.__batch_size})
        multi_evaluation_run_logger.info(f"starting evaluation on {len(evaluation_datasets)} datasets...")
        
        experiment_tasks: List[Coroutine[Any, Any, Tuple[str, Dict[str, float]]]] = [self.__evaluate(model, dataset, evaluation_metrics, multi_evaluation_run_logger) for dataset in evaluation_datasets.items()]

        completed, pending = await asyncio.wait(experiment_tasks)

        results = dict([t.result() for t in completed])

        multi_evaluation_run_logger.log(FINISHED_MULTI_DATASET_EVALUATION, {"model": model, "evaluation_datasets": evaluation_datasets, "evaluation_metrics": evaluation_metrics, "results": results, "batch_size": self.__batch_size})
        multi_evaluation_run_logger.info(f"finished evaluation on {len(evaluation_datasets)} datasets.")

        return results