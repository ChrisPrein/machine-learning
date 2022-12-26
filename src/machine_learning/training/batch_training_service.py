from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from logging import Logger
import logging
from typing import Deque, Generic, List, Optional, Dict, Tuple, TypeGuard, TypeVar, Union, overload
from tqdm import tqdm
import time
from custom_operators.operators.true_division import *
from ..modeling.model import TInput, TTarget
from .training_service import DATASET, TRAINING_DATASET, TModel, TrainingService

@dataclass
class TrainingContext(Generic[TInput, TTarget, TModel]):
    model: TModel
    dataset_name: str
    current_epoch: int
    current_batch_index: int
    train_losses: Deque[Union[float, Dict[str, float]]]
    continue_training: bool

class BatchTrainingPlugin(Generic[TInput, TTarget, TModel], ABC):
    pass

class PreLoop(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_loop(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        pass

class PostLoop(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_loop(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        pass

class PreEpoch(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_epoch(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        pass

class PostEpoch(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_epoch(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        pass

class PreTrain(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_train(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        pass

class PostTrain(BatchTrainingPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_train(self, logger: Logger, trainingContext: TrainingContext[TInput, TTarget, TModel]):
        pass

def is_batch(val: List[object]) -> TypeGuard[Tuple]:
    return all(isinstance(x, Tuple) for x in val)

def is_dataset(val: List[object]) -> TypeGuard[DATASET]:
    return all(is_batch(x) for x in val)

def is_list_dataset(val: List[object]) -> TypeGuard[List[DATASET]]:
    return all(is_dataset(x) for x in val)

class BatchTrainingService(TrainingService[TInput, TTarget, TModel], ABC):
    def __init__(self, logger: Optional[Logger]=None,
        max_epochs: int = 100, 
        max_losses: Optional[int] = None, 
        plugins: Dict[str, BatchTrainingPlugin[TInput, TTarget, TModel]] = {}, 
        **kwargs):
        
        self.__logger = logger if not logger is None else logging.getLogger()
        
        self.__max_epochs: int = max_epochs

        self.__max_losses: Optional[int] = max_losses
       
        self.__pre_loop_plugins: Dict[str, PreLoop[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PreLoop), plugins.items()))
        self.__post_loop_plugins: Dict[str, PostLoop[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PostLoop), plugins.items()))
        self.__pre_epoch_plugins: Dict[str, PreEpoch[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PreEpoch), plugins.items()))
        self.__post_epoch_plugins: Dict[str, PostEpoch[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PostEpoch), plugins.items()))
        self.__pre_train_plugins: Dict[str, PreTrain[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PreTrain), plugins.items()))
        self.__post_train_plugins: Dict[str, PostTrain[TInput, TTarget, TModel]] = dict(filter(lambda plugin: isinstance(plugin[1], PostTrain), plugins.items()))

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
    async def train(self, model: TModel, dataset: TRAINING_DATASET,  logger: Optional[Logger] = None) -> TModel:
        if isinstance(dataset, tuple):
            return await self.__train(model=model, training_dataset=dataset, logger=logger)
        else:
            return await self.__train(model=model, training_dataset=('dataset', dataset), logger=logger)

    async def __train(self, model: TModel, training_dataset: Tuple[str, DATASET],  logger: Optional[Logger] = None) -> TModel:
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