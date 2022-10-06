from abc import ABC
from collections import deque
from logging import Logger
import logging
from typing import List, Optional, Dict, Tuple, Union
import asyncio
import asyncio.tasks
import asyncio.futures
from dataset_handling.dataloader import DataLoader
from torch.utils.data import Dataset, random_split
import nest_asyncio
from tqdm import tqdm
import time

from machine_learning.training.abstractions.batch_training_plugin import BatchTrainingPlugin, PostEpoch, PostLoop, PostMultiLoop, PostTrain, PreEpoch, PreLoop, PreMultiLoop, PreTrain
from custom_operators.operators.true_division import *

from ..evaluation.abstractions.evaluation_service import EvaluationService
from ..modeling.abstractions.model import Model, TInput, TTarget
from .abstractions.training_service import DATASET, TRAINING_DATASET, TrainingService
from ..evaluation.abstractions.evaluation_metric import EvaluationContext, EvaluationMetric
from ..evaluation.default_evaluation_service import DefaultEvaluationService

from .abstractions.batch_training_plugin import *

nest_asyncio.apply()

def is_batch(val: List[object]) -> TypeGuard[Tuple]:
    return all(isinstance(x, Tuple) for x in val)

def is_dataset(val: List[object]) -> TypeGuard[DATASET]:
    return all(is_batch(x) for x in val)

def is_list_dataset(val: List[object]) -> TypeGuard[List[DATASET]]:
    return all(is_dataset(x) for x in val)

class BatchTrainingService(TrainingService[TInput, TTarget, TModel], ABC):
    def __init__(self, logger: Optional[Logger]=None, evaluation_service: Optional[EvaluationService[TInput, TTarget, TModel]] = None, 
    batch_size: int = 1, drop_last: bool = True, event_loop: Optional[asyncio.AbstractEventLoop] = None, max_epochs: int = 100, 
    training_dataset_size_ratio: float = 0.8, max_losses: Optional[int] = None, max_scores: Optional[int] = None, plugins: Dict[str, BatchTrainingPlugin[TInput, TTarget, TModel]] = {}, **kwargs):
        self.__logger = logger if not logger is None else logging.getLogger()
        
        if evaluation_service is None:
            evaluation_service = DefaultEvaluationService[TInput, TTarget, TModel](logger=self.__logger, batch_size=batch_size, drop_last=drop_last, event_loop=event_loop)
        
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()
        self.__max_epochs: int = max_epochs
        self.__batch_size: int = batch_size
        self.__drop_last: bool = drop_last
        self.__evaluation_service: EvaluationService[TInput, TTarget, TModel, EvaluationContext[TInput, TTarget, TModel]] = evaluation_service
        self.__training_dataset_size_ratio: float = training_dataset_size_ratio

        self.__max_losses: Optional[int] = max_losses
        self.__max_scores: Optional[int] = max_scores

        self.__pre_multi_loop_plugins: Dict[str, PreMultiLoop[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PreMultiLoop), plugins.items()))
        self.__post_multi_loop_plugins: Dict[str, PostMultiLoop[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PostMultiLoop), plugins.items()))
        self.__pre_loop_plugins: Dict[str, PreLoop[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PreLoop), plugins.items()))
        self.__post_loop_plugins: Dict[str, PostLoop[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PostLoop), plugins.items()))
        self.__pre_epoch_plugins: Dict[str, PreEpoch[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PreEpoch), plugins.items()))
        self.__post_epoch_plugins: Dict[str, PostEpoch[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PostEpoch), plugins.items()))
        self.__pre_train_plugins: Dict[str, PreTrain[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PreTrain), plugins.items()))
        self.__post_train_plugins: Dict[str, PostTrain[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PostTrain), plugins.items()))
        self.__pre_multi_train_step_plugins: Dict[str, PreMultiTrainStep] = dict(filter(lambda plugin: isinstance(plugin[1], PreMultiTrainStep), plugins.items()))
        self.__post_multi_train_step_plugins: Dict[str, PostMultiTrainStep] = dict(filter(lambda plugin: isinstance(plugin[1], PostMultiTrainStep), plugins.items()))

    def __execute_pre_multi_loop_plugins(self, logger: Logger, context: MultiTrainingContext):
        logger.debug("Executing pre multi loop plugins...")
        for name, plugin in self.__pre_multi_loop_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.pre_multi_loop(logger, context)

    def __execute_post_multi_loop_plugins(self, logger: Logger, context: MultiTrainingContext):
        logger.debug("Executing post multi loop plugins...")
        for name, plugin in self.__post_multi_loop_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.post_multi_loop(logger, context)

    def __execute_pre_multi_train_step_plugins(self, logger: Logger, context: MultiTrainingContext):
        logger.debug("Executing pre multi train step plugins...")
        for name, plugin in self.__pre_multi_train_step_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.pre_multi_train_step(logger, context)

    def __execute_post_multi_train_step_plugins(self, logger: Logger, context: MultiTrainingContext):
        logger.debug("Executing post multi train step plugins...")
        for name, plugin in self.__post_multi_train_step_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.post_multi_train_step(logger, context)

    def __execute_pre_loop_plugins(self, logger: Logger, context: TrainingContext[TInput, TTarget, TModel]):
        logger.debug("Executing pre loop plugins...")
        for name, plugin in self.__pre_loop_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.pre_loop(logger, context)

    def __execute_post_loop_plugins(self, logger: Logger, context: TrainingContext[TInput, TTarget, TModel]):
        logger.debug("Executing post loop plugins...")
        for name, plugin in self.__post_loop_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.post_loop(logger, context)

    def __execute_pre_epoch_plugins(self, logger: Logger, context: TrainingContext[TInput, TTarget, TModel]):
        logger.debug("Executing pre epoch plugins...")
        for name, plugin in self.__pre_epoch_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.pre_epoch(logger, context)

    def __execute_post_epoch_plugins(self, logger: Logger, context: TrainingContext[TInput, TTarget, TModel]):
        logger.debug("Executing post epoch plugins...")
        for name, plugin in self.__post_epoch_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.post_epoch(logger, context)

    def __execute_pre_train_plugins(self, logger: Logger, context: TrainingContext[TInput, TTarget, TModel]):
        logger.debug("Executing pre train plugins...")
        for name, plugin in self.__pre_train_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.pre_train(logger, context)

    def __execute_post_train_plugins(self, logger: Logger, context: TrainingContext[TInput, TTarget, TModel]):
        logger.debug("Executing post train plugins...")
        for name, plugin in self.__post_train_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.post_train(logger, context)

    def __has_max_number_of_epochs_been_reached(self, training_context: TrainingContext[TInput, TTarget, TModel]) -> bool:
        return training_context.current_epoch >= self.__max_epochs

    @overload
    async def train(self, model: TModel, dataset: DATASET, logger: Optional[Logger] = None) -> TModel: ...
    @overload
    async def train(self, model: TModel, dataset: Tuple[str, DATASET], logger: Optional[Logger] = None) -> TModel: ...
    @overload
    async def train(self, model: TModel, dataset: Iterable[DATASET], logger: Optional[Logger] = None) -> TModel: ...
    @overload
    async def train(self, model: TModel, dataset: Dict[str, DATASET], logger: Optional[Logger] = None) -> TModel: ...

    async def train(self, model: TModel, dataset: TRAINING_DATASET,  logger: Optional[Logger] = None) -> TModel:
        if isinstance(dataset, dict):
            return await self.__train_on_multiple_datasets(model=model, datasets=dataset, logger=logger)
        elif is_list_dataset(dataset):
            evaluation_datasets_with_names: Dict[str, DATASET] = {f'dataset_{index}': dataset for index, dataset in enumerate(dataset)}

            return await self.__train_on_multiple_datasets(model=model, datasets=evaluation_datasets_with_names, logger=logger)
        elif isinstance(dataset, tuple):
            return await self.__train_on_multiple_datasets(model=model, datasets={dataset[0]: dataset[1]}, logger=logger)
        else:
            return await self.__train_on_multiple_datasets(model=model, datasets={'dataset', dataset}, logger=logger)

    async def __train(self, model: TModel, training_dataset: TRAINING_DATASET,  logger: Optional[Logger] = None) -> TModel:
        logger = logger if not logger is None else self.__logger
        
        if model is None:
            raise ValueError("model")

        if training_dataset is None:
            raise ValueError("dataset")

        training_context: TrainingContext[TInput, TTarget, TModel] = None

        dataset: DATASET = training_dataset[1]
        dataset_name: str = training_dataset[0]

        training_context = TrainingContext[TInput, TTarget, TModel](model=model, dataset_name=dataset_name, train_losses=deque([], self.__max_losses), current_epoch=0, current_batch_index=0, continue_training=True)

        logger.info('Starting training loop...')

        self.__execute_pre_loop_plugins(logger, training_context)

        while training_context.continue_training and not self.__has_max_number_of_epochs_been_reached(training_context=training_context):
            logger.info("Starting epoch...")
            epoch_start_time: float = time.time()
            training_context.current_epoch += 1

            self.__execute_pre_epoch_plugins(logger, training_context)

            sum_iteration_run_time: float = 0
            count_iteration_run_times: int = 0

            sum_batch_load_time: float = 0
            count_batch_load_times: int = 0

            iteration_start_time: float = 0
            iteration_end_time: float = 0
            batch_load_start_time: float = 0

            batch_load_start_time = time.time()

            for batch_index, batch in enumerate(tqdm(dataset, miniters=len(dataset)/100, initial=training_context.current_batch_index)):
                training_context.current_batch_index = batch_index

                iteration_start_time = time.time()

                sum_batch_load_time += iteration_start_time - batch_load_start_time
                count_batch_load_times += 1

                logger.debug(f"Batch load took {iteration_start_time - batch_load_start_time} seconds.")

                self.__execute_pre_train_plugins(logger, training_context)

                inputs: List[TInput] = [value[0] for value in batch]
                targets: List[TTarget] = [value[1] for value in batch]

                logger.debug("Executing training step.")
                train_loss: Union[float, Dict[str, float]] = model.training_step(inputs, targets)
                
                training_context.train_losses.append(train_loss)

                self.__execute_post_train_plugins(logger, training_context)

                iteration_end_time = time.time()
                sum_iteration_run_time += iteration_end_time - iteration_start_time
                count_iteration_run_times += 1

                logger.debug(f"Iteration took {iteration_end_time - iteration_start_time} seconds.")

                batch_load_start_time = time.time()

            logger.info(f"Each batch load took around {sum_batch_load_time/allow_zero/count_batch_load_times} seconds.")
            logger.info(f"Each iteration took around {sum_iteration_run_time/allow_zero/count_iteration_run_times} seconds.")

            self.__execute_post_epoch_plugins(logger, training_context)

            logger.info("Finished epoch.")
            logger.info(f"Epoch took {time.time() - epoch_start_time} seconds.")

        logger.info("Finished training loop.")

        self.__execute_post_loop_plugins(logger, training_context)

        return model

    async def __train_on_multiple_datasets(self, model: TModel, datasets: Dict[str, DATASET], logger: Optional[Logger] = None) -> TModel:
        if logger == None:
            logger = self.__logger

        logger.info(f"Starting training on {len(datasets)} datasets...")

        context: MultiTrainingContext = MultiTrainingContext(current_dataset_index=0)

        self.__execute_pre_multi_loop_plugins(logger, context)

        for dataset_index in range(context.current_dataset_index, len(datasets.items())):
            context.current_dataset_index = dataset_index

            self.__execute_pre_multi_train_step_plugins(logger, context)

            dataset_name, dataset = list(datasets.items())[dataset_index]

            training_run_logger: Logger = logger.getChild(dataset_name)
            model = await self.__train(model, (dataset_name, dataset), training_run_logger)

            self.__execute_post_multi_train_step_plugins(logger, context)

        logger.info(f"Finished training on {len(datasets.items())} datasets.")

        self.__execute_post_multi_loop_plugins(logger, context)

        return model