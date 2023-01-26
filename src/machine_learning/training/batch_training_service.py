from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from logging import Logger
import logging
from typing import Deque, Generic, List, Optional, Dict, Tuple, TypeGuard, TypeVar, Union
from tqdm import tqdm
import time
from custom_operators.operators.true_division import *

from ..evaluation.evaluation_context import Prediction
from .trainer import Trainer
from ..modeling.model import TInput, TTarget, TOutput
from .training_service import Dataset, TrainingDataset, TModel, TrainingService

__all__ = ['TrainingContext', 'BatchTrainingPlugin', 'PreLoop', 'PostLoop', 'PreEpoch', 'PostEpoch', 'PreTrain', 'PostTrain', 'BatchTrainingService', 'TTrainer']

TTrainer = TypeVar('TTrainer', bound=Trainer)

@dataclass
class TrainingContext(Generic[TInput, TTarget, TOutput, TModel, TTrainer]):
    model: TModel
    trainer: TTrainer
    dataset_name: str
    current_epoch: int
    current_batch_index: int
    predictions: Deque[Prediction[TInput, TTarget, TOutput]]
    train_losses: Deque[Union[float, Dict[str, float]]]
    continue_training: bool

class BatchTrainingPlugin(Generic[TInput, TTarget, TOutput, TModel, TTrainer], ABC):
    pass

class PreLoop(BatchTrainingPlugin[TInput, TTarget, TOutput, TModel, TTrainer]):
    @abstractmethod
    def pre_loop(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        pass

class PostLoop(BatchTrainingPlugin[TInput, TTarget, TOutput, TModel, TTrainer]):
    @abstractmethod
    def post_loop(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        pass

class PreEpoch(BatchTrainingPlugin[TInput, TTarget, TOutput, TModel, TTrainer]):
    @abstractmethod
    def pre_epoch(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        pass

class PostEpoch(BatchTrainingPlugin[TInput, TTarget, TOutput, TModel, TTrainer]):
    @abstractmethod
    def post_epoch(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        pass

class PreTrain(BatchTrainingPlugin[TInput, TTarget, TOutput, TModel, TTrainer]):
    @abstractmethod
    def pre_train(self, logger: Logger, trainging_context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        pass

class PostTrain(BatchTrainingPlugin[TInput, TTarget, TOutput, TModel, TTrainer]):
    @abstractmethod
    def post_train(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        pass

def is_batch(val: List[object]) -> TypeGuard[Tuple]:
    return all(isinstance(x, Tuple) for x in val)

def is_dataset(val: List[object]) -> TypeGuard[Dataset]:
    return all(is_batch(x) for x in val)

def is_list_dataset(val: List[object]) -> TypeGuard[List[Dataset]]:
    return all(is_dataset(x) for x in val)

class BatchTrainingService(Generic[TInput, TTarget, TOutput, TModel, TTrainer], TrainingService[TInput, TTarget, TModel]):
    def __init__(self, 
        trainer: TTrainer, 
        logger: Optional[Logger]=None,
        max_epochs: int = 100, 
        max_losses: Optional[int] = None, 
        max_predictions: Optional[int] = None, 
        plugins: Dict[str, BatchTrainingPlugin[TInput, TTarget, TOutput, TModel, TTrainer]] = {}, 
        **kwargs):

        if trainer == None:
            raise TypeError('trainer')
        
        self.__logger = logger if not logger is None else logging.getLogger()
        self.__trainer: TTrainer = trainer
        self.__max_epochs: int = max_epochs
        self.__max_losses: Optional[int] = max_losses
        self.__max_predictions: Optional[int] = max_predictions
       
        self.__pre_loop_plugins: Dict[str, PreLoop[TInput, TTarget, TOutput, TModel, TTrainer]] = dict(filter(lambda plugin: isinstance(plugin[1], PreLoop), plugins.items()))
        self.__post_loop_plugins: Dict[str, PostLoop[TInput, TTarget, TOutput, TModel, TTrainer]] = dict(filter(lambda plugin: isinstance(plugin[1], PostLoop), plugins.items()))
        self.__pre_epoch_plugins: Dict[str, PreEpoch[TInput, TTarget, TOutput, TModel, TTrainer]] = dict(filter(lambda plugin: isinstance(plugin[1], PreEpoch), plugins.items()))
        self.__post_epoch_plugins: Dict[str, PostEpoch[TInput, TTarget, TOutput, TModel, TTrainer]] = dict(filter(lambda plugin: isinstance(plugin[1], PostEpoch), plugins.items()))
        self.__pre_train_plugins: Dict[str, PreTrain[TInput, TTarget, TOutput, TModel, TTrainer]] = dict(filter(lambda plugin: isinstance(plugin[1], PreTrain), plugins.items()))
        self.__post_train_plugins: Dict[str, PostTrain[TInput, TTarget, TOutput, TModel, TTrainer]] = dict(filter(lambda plugin: isinstance(plugin[1], PostTrain), plugins.items()))

    def __execute_pre_loop_plugins(self, logger: Logger, context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        logger.debug("Executing pre loop plugins...")
        for name, plugin in self.__pre_loop_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.pre_loop(logger, context)

    def __execute_post_loop_plugins(self, logger: Logger, context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        logger.debug("Executing post loop plugins...")
        for name, plugin in self.__post_loop_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.post_loop(logger, context)

    def __execute_pre_epoch_plugins(self, logger: Logger, context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        logger.debug("Executing pre epoch plugins...")
        for name, plugin in self.__pre_epoch_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.pre_epoch(logger, context)

    def __execute_post_epoch_plugins(self, logger: Logger, context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        logger.debug("Executing post epoch plugins...")
        for name, plugin in self.__post_epoch_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.post_epoch(logger, context)

    def __execute_pre_train_plugins(self, logger: Logger, context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        logger.debug("Executing pre train plugins...")
        for name, plugin in self.__pre_train_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.pre_train(logger, context)

    def __execute_post_train_plugins(self, logger: Logger, context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        logger.debug("Executing post train plugins...")
        for name, plugin in self.__post_train_plugins.items():
            logger.debug(f"Executing plugin with name {name}...")
            plugin.post_train(logger, context)

    def __has_max_number_of_epochs_been_reached(self, training_context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]) -> bool:
        return training_context.current_epoch >= self.__max_epochs

    async def train(self, model: TModel, dataset: TrainingDataset,  logger: Optional[Logger] = None) -> TModel:
        if isinstance(dataset, tuple):
            return await self.__train(model=model, training_dataset=dataset, logger=logger)
        else:
            return await self.__train(model=model, training_dataset=('dataset', dataset), logger=logger)

    async def __train(self, model: TModel, training_dataset: Tuple[str, Dataset],  logger: Optional[Logger] = None) -> TModel:
        logger = logger if not logger is None else self.__logger
        
        if model is None:
            raise ValueError("model")

        if training_dataset is None:
            raise ValueError("dataset")

        training_context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer] = None

        dataset: Dataset = training_dataset[1]
        dataset_name: str = training_dataset[0]

        training_context = TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer](model=model, trainer=self.__trainer, dataset_name=dataset_name, train_losses=deque([], self.__max_losses), predictions=deque([], self.__max_predictions), current_epoch=0, current_batch_index=0, continue_training=True)

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
                predictions, train_loss = self.__trainer(model, inputs, targets, logger)
                
                training_context.train_losses.append(train_loss)
                training_context.predictions.extend(predictions)

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