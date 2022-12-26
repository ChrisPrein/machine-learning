from collections import deque
from logging import Logger
import logging
from typing import Callable, Iterable, List, Optional, Dict, Tuple, TypeGuard, Union, overload
import time

from .evaluator import EVALUATOR_RESULT, INPUT, TARGET, Evaluator
from .multi_metric import MultiMetric
from .evaluation_metric import EvaluationMetric
from .multi_evaluation_context import MultiEvaluationContext, Score
from .default_evaluation_plugin import DefaultEvaluationPlugin, PostEvaluationStep, PostLoop, PostMultiEvaluationStep, PostMultiLoop, PreEvaluationStep, PreLoop, PreMultiEvaluationStep, PreMultiLoop
from .evaluation_context import EvaluationContext, Prediction, TModel
from .evaluation_service import DATASET, EVALUATION_DATASET, EVALUATION_METRICS, EVALUATION_RESULT, PREDICTION_DATA, PREDICTIONS, EvaluationService
from ..modeling.model import TInput, TTarget
from custom_operators.operators.true_division import *
from tqdm import tqdm
import nest_asyncio

nest_asyncio.apply()

def is_batch(val: List[object]) -> TypeGuard[Tuple]:
    return all(isinstance(x, Tuple) for x in val)

def is_dataset(val: List[object]) -> TypeGuard[DATASET]:
    return all(is_batch(x) for x in val)

def is_list_dataset(val: List[object]) -> TypeGuard[List[DATASET]]:
    return all(is_dataset(x) for x in val)

class DefaultEvaluationService(EvaluationService[TInput, TTarget, TModel]):
    def __init__(self, 
        evaluator: Callable[[TModel, INPUT, TARGET, Logger], EVALUATOR_RESULT],
        logger: Optional[Logger]=None, plugins: Dict[str, DefaultEvaluationPlugin[TInput, TTarget, TModel]] = {}, 
        max_predictions: Optional[int] = None, max_losses: Optional[int] = None, **kwargs):

        if evaluator == None:
            raise TypeError("evaluator")

        self.__logger = logger if not logger is None else logging.getLogger()
        self.__evaluator: Callable[[TModel, INPUT, TARGET, Logger], EVALUATOR_RESULT] = evaluator
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

    def __predict_batch(self, evaluation_context: EvaluationContext[TInput, TTarget, TModel], model: TModel, batch: List[Tuple[TInput, TTarget]], logger: Logger) -> Tuple[List[Prediction], Union[float, Dict[str, float]]]:
        inputs: List[TInput] = [sample[0] for sample in batch]
        targets: List[TTarget] = [sample[1] for sample in batch]
        predictions, loss = self.__evaluator(model, inputs, targets, logger)

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

    @overload
    async def evaluate(self, model: TModel, evaluation_dataset: DATASET, evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> Dict[str, Score]: ...
    @overload
    async def evaluate(self, model: TModel, evaluation_dataset: Tuple[str, DATASET], evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> Dict[str, Score]: ...
    @overload
    async def evaluate(self, model: TModel, evaluation_dataset: Dict[str, DATASET], evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> Dict[str, Dict[str, Score]]: ...
    @overload
    async def evaluate(self, model: TModel, evaluation_dataset: Iterable[DATASET], evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> Dict[str, Dict[str, Score]]: ...

    async def evaluate(self, model: TModel, evaluation_dataset: EVALUATION_DATASET, evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> EVALUATION_RESULT:
        if isinstance(evaluation_dataset, dict):
            return await self.__evaluate_on_multiple_datasets(model=model, evaluation_datasets=evaluation_dataset, evaluation_metrics=evaluation_metrics, logger=logger)
        elif is_list_dataset(evaluation_dataset):
            evaluation_datasets_with_names: Dict[str, DATASET] = {f'dataset_{index}': dataset for index, dataset in enumerate(evaluation_dataset)}

            return await self.__evaluate_on_multiple_datasets(model=model, evaluation_datasets=evaluation_datasets_with_names, evaluation_metrics=evaluation_metrics, logger=logger)
        elif isinstance(evaluation_dataset, tuple):
            return list((await self.__evaluate_on_multiple_datasets(model=model, evaluation_datasets={evaluation_dataset[0]: evaluation_dataset[1]}, evaluation_metrics=evaluation_metrics, logger=logger)).values())[0]
        else:
            return list((await self.__evaluate_on_multiple_datasets(model=model, evaluation_datasets={'dataset': evaluation_dataset}, evaluation_metrics=evaluation_metrics, logger=logger)).values())[0]

    @overload
    async def evaluate_predictions(self, predictions: PREDICTIONS, evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> Dict[str, Score]: ...
    @overload
    async def evaluate_predictions(self, predictions: Tuple[str, PREDICTIONS], evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> Dict[str, Score]: ...
    
    async def evaluate_predictions(self, predictions: PREDICTION_DATA, evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> Dict[str, Score]:
        if isinstance(predictions, tuple):
            return await self.__evaluate_predictions(predictions=predictions, evaluation_metrics=evaluation_metrics, logger=logger)
        else:
            return await self.__evaluate_predictions(predictions=('dataset', predictions), evaluation_metrics=evaluation_metrics, logger=logger)

    async def __evaluate(self, model: TModel, evaluation_dataset: Tuple[str, DATASET], evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> Dict[str, Score]:
        if logger is None:
            logger = self.__logger
        
        if model is None:
            raise ValueError("model")

        if evaluation_dataset is None:
            raise ValueError("evaluation_dataset")

        if evaluation_metrics is None:
            raise ValueError("evaluation_metrics")

        dataset: DATASET = evaluation_dataset[1]
        dataset_name: str = evaluation_dataset[0]

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
            
            predictions, loss = self.__predict_batch(evaluation_context, model, batch, logger)

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

    async def __evaluate_on_multiple_datasets(self, model: TModel, evaluation_datasets: Dict[str, DATASET], evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> Dict[str, Dict[str, Score]]:
        if logger == None:
            logger = self.__logger

        logger.info(f"starting evaluation on {len(evaluation_datasets)} datasets...")

        context: MultiEvaluationContext = MultiEvaluationContext(current_dataset_index=0, scores={})

        self.__execute_pre_multi_loop_plugins(logger, context)

        for dataset_index in range(context.current_dataset_index, len(evaluation_datasets)):
            context.current_dataset_index = dataset_index

            self.__execute_pre_multi_evaluation_step_plugins(logger, context)

            dataset_name, dataset = list(evaluation_datasets.items())[dataset_index]

            evaluation_logger: Logger = logger.getChild(dataset_name)
            context.scores[dataset_name] = await self.__evaluate(model, (dataset_name, dataset), evaluation_metrics, evaluation_logger)

            self.__execute_post_multi_evaluation_step_plugins(logger, context)

        logger.info(f"finished evaluation on {len(evaluation_datasets)} datasets.")

        self.__execute_post_multi_loop_plugins(logger, context)

        return context.scores

    async def __evaluate_predictions(self, predictions: Tuple[str, PREDICTIONS], evaluation_metrics: EVALUATION_METRICS, logger: Optional[Logger] = None) -> Dict[str, Score]:
        if logger is None:
            logger = self.__logger
        
        if predictions is None:
            raise ValueError("predictions")

        if evaluation_metrics is None:
            raise ValueError("evaluation_metrics")

        prediction_set: PREDICTIONS = predictions[1]
        dataset_name: str = predictions[0]

        evaluation_context: EvaluationContext[TInput, TTarget, TModel] = EvaluationContext[TInput, TTarget, TModel](None, dataset_name, deque([], self.__max_predictions), 0, deque([], self.__max_losses))

        self.__reset_evaluation_metrics(logger, evaluation_metrics)

        logger.info('Starting evaluation loop...')
        evaluation_start_time: float = time.time()

        self.__execute_pre_loop_plugins(logger, evaluation_context)

        sum_iteration_run_time: float = 0
        count_iteration_run_times: int = 0

        iteration_start_time: float = 0
        iteration_end_time: float = 0

        for prediction_index, prediction in enumerate(tqdm(prediction_set, miniters=len(prediction_set)/100, initial=evaluation_context.current_batch_index)):
            evaluation_context.current_batch_index = prediction_index

            iteration_start_time = time.time()

            self.__execute_pre_evaluation_step_plugins(logger, evaluation_context)
            
            evaluation_context.predictions.append(prediction)

            self.__update_evaluation_metrics(logger, evaluation_metrics, [prediction])

            self.__execute_post_evaluation_step_plugins(logger, evaluation_context)

            iteration_end_time = time.time()
            sum_iteration_run_time += iteration_end_time - iteration_start_time
            count_iteration_run_times += 1

            logger.debug(f"Iteration took {iteration_end_time - iteration_start_time} seconds.")

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