from collections import deque
from logging import Logger
import logging
from typing import List, Optional, Dict, Tuple, Union
import time
from ..evaluation.abstractions.default_evaluation_plugin import *
from ..modeling.abstractions.model import Model, TInput, TTarget
from .abstractions.evaluation_metric import *
from .abstractions.multi_metric import *
from .abstractions.evaluation_service import EvaluationService, Score
from ..operators.true_division import *
import asyncio
import asyncio.tasks
import asyncio.futures
from dataset_handling.dataloader import DataLoader
from tqdm import tqdm
import nest_asyncio

nest_asyncio.apply()

class DefaultEvaluationService(EvaluationService[TInput, TTarget, TModel]):
    def __init__(self, logger: Optional[Logger]=None, plugins: Dict[str, DefaultEvaluationPlugin[TInput, TTarget, TModel]] = {}, max_predictions: Optional[int] = None, max_losses: Optional[int] = None, **kwargs):
        self.__logger = logger if not logger is None else logging.getLogger()
        self.__max_predictions: Optional[int] = max_predictions
        self.__max_losses: Optional[int] = max_losses

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

    def __predict_batch(self, evaluation_context: EvaluationContext[TInput, TTarget, TModel], model: TModel, batch: List[Tuple[TInput, TTarget]]) -> Tuple[List[Prediction], Union[float, Dict[str, float]]]:
        inputs: List[TInput] = [sample[0] for sample in batch]
        targets: List[TTarget] = [sample[1] for sample in batch]
        predictions, loss = model.evaluation_step(inputs, targets)

        combined: List[Tuple[TInput, TTarget, TTarget]] = zip(inputs, predictions, targets)

        return [Prediction(result[0], result[1], result[2]) for result in combined], loss

    def __reset_evaluation_metrics(self, logger: Logger, evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]]):
        logger.debug("Reseting evaluation metrics...")
        for metric_name, evaluation_metrtic in evaluation_metrics.items():
            evaluation_metrtic.reset()

    def __update_evaluation_metrics(self, logger: Logger, evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]], batch: List[Prediction]):
        logger.debug("Updating evaluation metrics...")
        for metric_name, evaluation_metrtic in evaluation_metrics.items():
            evaluation_metrtic.update(batch)

    async def evaluate(self, model: TModel, evaluation_dataset: Union[Tuple[str, Iterable[Iterable[Tuple[TInput, TTarget]]]], Iterable[Iterable[Tuple[TInput, TTarget]]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]], logger: Optional[Logger] = None) -> Dict[str, Score]:
        if logger is None:
            logger = self.__logger
        
        if model is None:
            raise ValueError("model")

        if evaluation_dataset is None:
            raise ValueError("evaluation_dataset")

        if evaluation_metrics is None:
            raise ValueError("evaluation_metrics")

        dataset: Iterable[Iterable[Tuple[TInput, TTarget]]] = None
        dataset_name: str = None

        if isinstance(evaluation_dataset, Tuple):
            dataset = evaluation_dataset[1]
            dataset_name = evaluation_dataset[0]
        else:
            dataset = evaluation_dataset
            dataset_name = type(dataset).__name__

        evaluation_context: EvaluationContext[TInput, TTarget, TModel] = EvaluationContext[TInput, TTarget, TModel](model, dataset_name, deque([], self.__max_predictions), 0, deque([], self.__max_losses))

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

        for batch_index, batch in enumerate(tqdm(dataset, miniters=len(dataset)/100, initial=evaluation_context.current_batch_index)):
            evaluation_context.current_batch_index = batch_index

            iteration_start_time = time.time()

            sum_batch_load_time += iteration_start_time - batch_load_start_time
            count_batch_load_times += 1

            logger.debug(f"Batch load took {iteration_start_time - batch_load_start_time} seconds.")

            self.__execute_pre_evaluation_step_plugins(logger, evaluation_context)
            
            predictions, loss = self.__predict_batch(evaluation_context, model, batch)

            evaluation_context.predictions.extend(predictions)
            evaluation_context.losses.append(loss)

            self.__update_evaluation_metrics(logger, evaluation_metrics, predictions)

            self.__execute_post_evaluation_step_plugins(logger, evaluation_context)

            iteration_end_time = time.time()
            sum_iteration_run_time += iteration_end_time - iteration_start_time
            count_iteration_run_times += 1

            logger.debug(f"Iteration took {iteration_end_time - iteration_start_time} seconds.")

            batch_load_start_time = time.time()

        logger.info(f"Each batch load took around {sum_batch_load_time /allow_zero/ count_batch_load_times} seconds.")
        logger.info(f"Each iteration took around {sum_iteration_run_time /allow_zero/ count_iteration_run_times} seconds.")

        result: Dict[str, Score] = {}

        for metric_name, metric in evaluation_metrics.items():
            if isinstance(metric, MultiMetric):
                current_scores = {name: Score(value, f'{metric_name}/{name}', dataset_name) for name, value in metric.scores.items()}
                result.update(current_scores)
            else:
                result[metric_name] = Score(metric.score, metric_name, dataset_name)

        logger.info('Finished evaluation loop.')
        logger.info(f"Epoch took {time.time() - evaluation_start_time} seconds.")

        self.__execute_post_loop_plugins(logger, evaluation_context, result)
        
        return result

    async def evaluate_on_multiple_datasets(self, model: TModel, evaluation_datasets: Dict[str, Iterable[Iterable[Tuple[TInput, TTarget]]]], evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget]]) -> Dict[str, Dict[str, Score]]:
        self.__logger.info(f"starting evaluation on {len(evaluation_datasets)} datasets...")

        context: MultiEvaluationContext = MultiEvaluationContext(current_dataset_index=0, scores={})

        self.__execute_pre_multi_loop_plugins(self.__logger, context)

        for dataset_index in range(context.current_dataset_index, len(evaluation_datasets)):
            context.current_dataset_index = dataset_index

            self.__execute_pre_multi_evaluation_step_plugins(self.__logger, context)

            dataset_name, dataset = list(evaluation_datasets.items())[dataset_index]

            evaluation_logger: Logger = self.__logger.getChild(dataset_name)
            context.scores[dataset_name] = await self.evaluate(model, (dataset_name, dataset), evaluation_metrics, evaluation_logger)

            self.__execute_post_multi_evaluation_step_plugins(self.__logger, context)

        self.__logger.info(f"finished evaluation on {len(evaluation_datasets)} datasets.")

        self.__execute_post_multi_loop_plugins(self.__logger, context)

        return context.scores