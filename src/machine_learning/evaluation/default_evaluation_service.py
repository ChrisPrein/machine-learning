from collections import deque
from logging import Logger
import logging
from typing import List, Optional, Dict, Tuple, Union
import time

from attr import asdict

from ..evaluation.abstractions.default_evaluation_plugin import *
from ..modeling.abstractions.model import Model, TInput, TTarget
from .abstractions.evaluation_metric import EvaluationContext, EvaluationMetric, Prediction, TModel
from .abstractions.evaluation_service import EvaluationService, Score
import asyncio
import asyncio.tasks
import asyncio.futures
from dataset_handling.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import nest_asyncio

nest_asyncio.apply()

class DefaultEvaluationService(EvaluationService[TInput, TTarget, TModel]):
    def __init__(self, logger: Optional[Logger]=None, batch_size: int = 1, drop_last: bool = True, 
    event_loop: Optional[asyncio.AbstractEventLoop] = None, plugins: Dict[str, DefaultEvaluationPlugin[TInput, TTarget, TModel]] = {}):
        self.__logger = logger if not logger is None else logging.getLogger()
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()
        self.__batch_size: int = batch_size
        self.__drop_last: bool = drop_last

        self.__pre_multi_loop_plugins: Dict[str, PreMultiLoop[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PreMultiLoop), plugins.items()))
        self.__post_multi_loop_plugins: Dict[str, PostMultiLoop[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PostMultiLoop), plugins.items()))
        self.__pre_loop_plugins: Dict[str, PreLoop[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PreLoop), plugins.items()))
        self.__post_loop_plugins: Dict[str, PostLoop[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PostLoop), plugins.items()))
        self.__pre_evaluation_step_plugins: Dict[str, PreEvaluationStep[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PreEvaluationStep), plugins.items()))
        self.__post_evaluation_step_plugins: Dict[str, PostEvaluationStep[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PostEvaluationStep), plugins.items()))
        self.__pre_multi_evaluation_step_plugins: Dict[str, PreMultiEvaluationStep[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PreMultiEvaluationStep), plugins.items()))
        self.__post_multi_evaluation_step_plugins: Dict[str, PostMultiEvaluationStep[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PostMultiEvaluationStep), plugins.items()))

    def __execute_pre_multi_loop_plugins(self, logger: Logger, context: MultiEvaluationContext):
        logger.debug("Executing pre multi loop plugins...")
        for name, plugin in self.__pre_multi_loop_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.pre_multi_loop(logger, context)

    def __execute_post_multi_loop_plugins(self, logger: Logger, context: MultiEvaluationContext):
        logger.debug("Executing post multi loop plugins...")
        for name, plugin in self.__post_multi_loop_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.post_multi_loop(logger, context)

    def __execute_pre_multi_evaluation_step_plugins(self, logger: Logger, context: MultiEvaluationContext):
        logger.debug("Executing pre multi train step plugins...")
        for name, plugin in self.__pre_multi_evaluation_step_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.pre_multi_evaluation_step(logger, context)

    def __execute_post_multi_evaluation_step_plugins(self, logger: Logger, context: MultiEvaluationContext):
        logger.debug("Executing post multi train step plugins...")
        for name, plugin in self.__post_multi_evaluation_step_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.post_multi_evaluation_step(logger, context)

    def __execute_pre_loop_plugins(self, logger: Logger, context: EvaluationContext[TInput, TTarget, TModel]):
        logger.debug("Executing pre loop plugins...")
        for name, plugin in self.__pre_loop_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.pre_loop(logger, context)

    def __execute_post_loop_plugins(self, logger: Logger, context: EvaluationContext[TInput, TTarget, TModel], result: Dict[str, Score]):
        logger.debug("Executing post loop plugins...")
        for name, plugin in self.__post_loop_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.post_loop(logger, context, result)

    def __execute_pre_evaluation_step_plugins(self, logger: Logger, context: EvaluationContext[TInput, TTarget, TModel]):
        logger.debug("Executing pre train plugins...")
        for name, plugin in self.__pre_evaluation_step_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.pre_evaluation_step(logger, context)

    def __execute_post_evaluation_step_plugins(self, logger: Logger, context: EvaluationContext[TInput, TTarget, TModel]):
        logger.debug("Executing post train plugins...")
        for name, plugin in self.__post_evaluation_step_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.post_evaluation_step(logger, context)

    def __predict_batch(self, evaluation_context: EvaluationContext[TInput, TTarget, TModel], model: TModel, batch: List[Tuple[TInput, TTarget]]) -> List[Prediction]:
        inputs: List[TInput] = [sample[0] for sample in batch]
        targets: List[TInput] = [sample[1] for sample in batch]
        predictions: List[TTarget] = model.predict_step(inputs)

        combined: List[Tuple[TInput, TTarget, TTarget]] = zip(inputs, predictions, targets)

        return [Prediction(result[0], result[1], result[2]) for result in combined]

    def __reset_evaluation_metrics(self, logger: Logger, evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]]):
        logger.debug("Reseting evaluation metrics...")
        for metric_name, evaluation_metrtic in evaluation_metrics.items():
            evaluation_metrtic.reset()

    def __update_evaluation_metrics(self, logger: Logger, evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]], batch: List[Prediction]):
        logger.debug("Updating evaluation metrics...")
        for metric_name, evaluation_metrtic in evaluation_metrics.items():
            evaluation_metrtic.update(batch)

    async def evaluate(self, model: TModel, evaluation_dataset: Union[Tuple[str, Dataset[Tuple[TInput, TTarget]]], Dataset[Tuple[TInput, TTarget]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]], logger: Optional[Logger] = None) -> Dict[str, Score]:
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

        evaluation_context: EvaluationContext[TInput, TTarget, TModel] = EvaluationContext[TInput, TTarget, TModel](model, dataset_name, deque([]), 0)

        data_loader: DataLoader[Tuple[TInput, TTarget]] = DataLoader[Tuple[TInput, TTarget]](dataset=dataset, batch_size=self.__batch_size, drop_last=self.__drop_last)

        self.__reset_evaluation_metrics(logger, evaluation_metrics)

        logger.info('Starting evaluation loop...')
        evaluation_start_time: float = time.time()

        self.__execute_pre_loop_plugins(logger, evaluation_context)

        sum_iteration_run_time: float = 0
        count_iteration_run_times: int = 0

        sum_batch_load_time: float = 0
        count_batch_load_times: int = 0

        iteration_start_time: float = 0
        iteration_end_time: float = 0
        batch_load_start_time: float = 0

        batch_load_start_time = time.time()

        for batch_index, batch in enumerate(tqdm(data_loader, miniters=len(dataset)/100, initial=evaluation_context.current_batch_index)):
            evaluation_context.current_batch_index = batch_index

            iteration_start_time = time.time()

            sum_batch_load_time += iteration_start_time - batch_load_start_time
            count_batch_load_times += 1

            logger.debug(f"Batch load took {iteration_start_time - batch_load_start_time} seconds.")

            self.__execute_pre_evaluation_step_plugins(logger, evaluation_context)
            
            predictions: List[Prediction] = self.__predict_batch(evaluation_context, model, batch)

            evaluation_context.predictions.extend(predictions)

            self.__update_evaluation_metrics(logger, evaluation_metrics, predictions)

            self.__execute_post_evaluation_step_plugins(logger, evaluation_context)

            iteration_end_time = time.time()
            sum_iteration_run_time += iteration_end_time - iteration_start_time
            count_iteration_run_times += 1

            logger.debug(f"Iteration took {iteration_end_time - iteration_start_time} seconds.")

            batch_load_start_time = time.time()

        logger.info(f"Each batch load took around {sum_batch_load_time/count_batch_load_times} seconds.")
        logger.info(f"Each iteration took around {sum_iteration_run_time/count_iteration_run_times} seconds.")

        result: Dict[str, Score] = {name: Score(evaluation_metric.score, name, dataset_name) for name, evaluation_metric in evaluation_metrics.items()}

        logger.info('Finished evaluation loop.')
        logger.info(f"Epoch took {time.time() - evaluation_start_time} seconds.")

        self.__execute_post_loop_plugins(logger, evaluation_context, result)
        
        return result

    async def __evaluate(self, model: TModel, evaluation_dataset: Tuple[str, Dataset[Tuple[TInput, TTarget]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]], logger: Logger) -> Tuple[str, Dict[str, Score]]:
        result = await self.evaluate(model, evaluation_dataset, evaluation_metrics, logger)

        return (evaluation_dataset[0], result)

    async def evaluate_on_multiple_datasets(self, model: TModel, evaluation_datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]]) -> Dict[str, Dict[str, Score]]:
        self.__logger.info(f"starting evaluation on {len(evaluation_datasets)} datasets...")

        context: MultiEvaluationContext = MultiEvaluationContext(current_dataset_index=0, scores={})

        self.__execute_pre_multi_loop_plugins(self.__logger, context)

        for dataset_index in range(context.current_dataset_index, evaluation_datasets.items()):
            context.current_dataset_index = dataset_index

            self.__execute_pre_multi_evaluation_step_plugins(self.__logger, context)

            dataset_name, dataset = evaluation_datasets.items()[0]

            evaluation_logger: Logger = self.__logger.getChild(dataset_name)
            context.scores[dataset_name] = self.evaluate(model, (dataset_name, dataset), evaluation_metrics, evaluation_logger)

            self.__execute_post_multi_evaluation_step_plugins(self.__logger, context)

        self.__logger.info(f"finished evaluation on {len(evaluation_datasets)} datasets.")

        self.__execute_post_multi_loop_plugins(self.__logger, context)

        return context.scores