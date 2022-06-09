from logging import Logger
import logging
from typing import Any, Callable, Coroutine, TypeVar, List, Generic, Optional, Dict, Tuple, Union
from uuid import UUID
import uuid
import time

from attr import asdict
from ..modeling.abstractions.model import Model, TInput, TTarget
from .abstractions.evaluation_metric import EvaluationContext, EvaluationMetric, Prediction, TModel
from .abstractions.evaluation_service import EvaluationService, Score
from .default_evaluation import default_evaluation
import asyncio
import asyncio.tasks
import asyncio.futures
from dataset_handling.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import nest_asyncio

nest_asyncio.apply()

class DefaultEvaluationService(EvaluationService[TInput, TTarget, TModel]):
    def __init__(self, pre_loop_hook: Optional[Callable[[Logger, EvaluationContext[TInput, TTarget, TModel]], None]] = None,
    pre_multi_loop_hook: Optional[Callable[[Logger], None]] = None, post_multi_loop_hook: Optional[Callable[[Logger], None]] = None,
    post_loop_hook: Optional[Callable[[Logger, EvaluationContext[TInput, TTarget, TModel], Dict[str, Score]], None]] = None, evaluation_hook: Callable[[Logger, EvaluationContext[TInput, TTarget, TModel], TModel, List[TInput], List[TTarget]], 
    List[TTarget]] = default_evaluation, logger: Optional[Logger]=None, batch_size: int = 1, drop_last: bool = True, event_loop: Optional[asyncio.AbstractEventLoop] = None):
        if evaluation_hook is None:
            raise ValueError("evaluation_hook")
        
        self.__logger = logger if not logger is None else logging.getLogger()
        self.__pre_multi_loop_hook: Optional[Callable[[Logger], None]] = pre_multi_loop_hook
        self.__post_multi_loop_hook: Optional[Callable[[Logger], None]] = post_multi_loop_hook
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()
        self.__batch_size: int = batch_size
        self.__drop_last: bool = drop_last
        self.__pre_loop_hook: Optional[Callable[[Logger, EvaluationContext[TInput, TTarget, TModel]], None]] = pre_loop_hook
        self.__post_loop_hook: Optional[Callable[[Logger, EvaluationContext[TInput, TTarget, TModel], Dict[str, Score]], None]] = post_loop_hook
        self.__evaluation_hook: Callable[[Logger, EvaluationContext[TInput, TTarget, TModel], TModel, List[TInput], List[TTarget]], List[TTarget]] = evaluation_hook

    def __predict_batch(self, evaluation_context: EvaluationContext[TInput, TTarget, TModel], model: TModel, batch: List[Tuple[TInput, TTarget]]) -> List[Prediction]:
        inputs: List[TInput] = [sample[0] for sample in batch]
        targets: List[TInput] = [sample[1] for sample in batch]
        predictions: List[TTarget] = self.__evaluation_hook(self.__logger, evaluation_context, model, inputs, targets)

        combined: List[Tuple[TInput, TTarget, TTarget]] = zip(inputs, predictions, targets)

        return [Prediction(result[0], result[1], result[2]) for result in combined]

    async def evaluate(self, model: TModel, evaluation_dataset: Union[Tuple[str, Dataset[Tuple[TInput, TTarget]]], Dataset[Tuple[TInput, TTarget]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget, TModel]], logger: Optional[Logger] = None) -> Dict[str, Score]:
        if logger is None:
            logger = self.__logger
        
        if model is None:
            raise ValueError("model")

        if evaluation_dataset is None:
            raise ValueError("evaluation_dataset")

        if evaluation_metrics is None:
            raise ValueError("evaluation_metrics")

        dataset: Dataset[Tuple[TInput, TTarget]] = None
        dataset_name: str = None

        if isinstance(evaluation_dataset, Tuple):
            dataset = evaluation_dataset[1]
            dataset_name = evaluation_dataset[0]
        else:
            dataset = evaluation_dataset
            dataset_name = type(dataset).__name__

        evaluation_context: EvaluationContext[TInput, TTarget, TModel] = EvaluationContext[TInput, TTarget, TModel](model, dataset_name, [])

        data_loader: DataLoader[Tuple[TInput, TTarget]] = DataLoader[Tuple[TInput, TTarget]](dataset=dataset, batch_size=self.__batch_size, drop_last=self.__drop_last)

        logger.info('Starting evaluation loop...')
        evaluation_start_time: float = time.time()

        if not self.__pre_loop_hook is None:
            logger.debug("Executing pre loop hook.")
            self.__pre_loop_hook(logger, evaluation_context)

        sum_iteration_run_time: float = 0
        count_iteration_run_times: int = 0

        sum_batch_load_time: float = 0
        count_batch_load_times: int = 0

        iteration_start_time: float = 0
        iteration_end_time: float = 0
        batch_load_start_time: float = 0

        batch_load_start_time = time.time()

        for batch_index, batch in enumerate(tqdm(data_loader, miniters=len(dataset)/100)):
            
            iteration_start_time = time.time()

            sum_batch_load_time += iteration_start_time - batch_load_start_time
            count_batch_load_times += 1

            logger.debug(f"Batch load took {iteration_start_time - batch_load_start_time} seconds.")
            
            predictions: List[Prediction] = self.__predict_batch(evaluation_context, model, batch)

            evaluation_context.predictions.extend(predictions)

            iteration_end_time = time.time()
            sum_iteration_run_time += iteration_end_time - iteration_start_time
            count_iteration_run_times += 1

            logger.debug(f"Iteration took {iteration_end_time - iteration_start_time} seconds.")

            batch_load_start_time = time.time()

        logger.info(f"Each batch load took around {sum_batch_load_time/count_batch_load_times} seconds.")
        logger.info(f"Each iteration took around {sum_iteration_run_time/count_iteration_run_times} seconds.")

        result: Dict[str, Score] = {}

        for i, (name, evaluation_metric) in enumerate(evaluation_metrics.items()):
            value: float = evaluation_metric.calculate_score(context=evaluation_context)

            result[name] = Score[TInput, TTarget](value, evaluation_context.predictions, name, dataset_name)

        logger.info('Finished evaluation loop.')
        logger.info(f"Epoch took {time.time() - evaluation_start_time} seconds.")

        if not self.__post_loop_hook is None:
            logger.debug("Executing post loop hook.")
            self.__post_loop_hook(logger, evaluation_context, result)
        
        return result

    async def __evaluate(self, model: TModel, evaluation_dataset: Tuple[str, Dataset[Tuple[TInput, TTarget]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget, TModel]], logger: Logger) -> Tuple[str, Dict[str, Score]]:
        result = await self.evaluate(model, evaluation_dataset, evaluation_metrics, logger)

        return (evaluation_dataset[0], result)

    async def evaluate_on_multiple_datasets(self, model: TModel, evaluation_datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget, TModel]]) -> Dict[str, Dict[str, Score]]:
        self.__logger.info(f"starting evaluation on {len(evaluation_datasets)} datasets...")

        if not self.__pre_multi_loop_hook is None:
            self.__logger.debug("Executing pre loop hook.")
            self.__pre_multi_loop_hook(self.__logger)
        
        experiment_tasks: List[Coroutine[Any, Any, Tuple[str, Dict[str, Score]]]] = [self.__evaluate(model, dataset, evaluation_metrics, self.__logger.getChild(str(dataset[0]))) for dataset in evaluation_datasets.items()]

        experiment_results: List[Tuple[str, Dict[str, Score]]] = await asyncio.gather(*experiment_tasks, loop=self.__event_loop)

        results = dict(experiment_results)

        self.__logger.info(f"finished evaluation on {len(evaluation_datasets)} datasets.")

        if not self.__post_multi_loop_hook is None:
            self.__logger.debug("Executing post loop hook.")
            self.__post_multi_loop_hook(self.__logger)

        return results