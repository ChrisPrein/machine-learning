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

from ..evaluation.abstractions.evaluation_service import EvaluationService
from .abstractions.objective_function import ObjectiveFunction
from .abstractions.stop_condition import StopCondition, TrainingContext, Score
from ..modeling.abstractions.model import Model, TInput, TTarget
from .abstractions.training_service import TrainingService
from ..evaluation.abstractions.evaluation_metric import EvaluationContext
from ..evaluation.default_evaluation_service import DefaultEvaluationService

from .abstractions.batch_training_plugin import *

nest_asyncio.apply()

class BatchTrainingService(TrainingService[TInput, TTarget, TModel], ABC):
    def __init__(self, logger: Optional[Logger]=None, evaluation_service: Optional[EvaluationService[TInput, TTarget, TModel]] = None, 
    batch_size: int = 1, drop_last: bool = True, event_loop: Optional[asyncio.AbstractEventLoop] = None, max_epochs: int = 100, 
    max_iterations: int = 10000, training_dataset_size_ratio: float = 0.8, max_losses: Optional[int] = None, max_scores: Optional[int] = None, plugins: Dict[str, BatchTrainingPlugin] = {}):
        self.__logger = logger if not logger is None else logging.getLogger()
        
        if evaluation_service is None:
            evaluation_service = DefaultEvaluationService[TInput, TTarget, TModel](logger=self.__logger, batch_size=batch_size, drop_last=drop_last, event_loop=event_loop)
        
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()
        self.__max_epochs: int = max_epochs
        self.__max_iterations: int = max_iterations
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

    def __is_any_stop_condition_satisfied(self, training_context: TrainingContext[TInput, TTarget, TModel], stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]], logger: Optional[Logger] = None) -> bool:
        if logger is None:
            logger = self.__logger
        
        self.__logger.info("Checking stop conditions...")
        is_any_satisfied: bool = False

        if training_context.current_epoch > self.__max_epochs: 
            self.__logger.info("Max number of epochs reached.")
            is_any_satisfied = True
        elif training_context.current_batch_index > self.__max_iterations:
            self.__logger.info("Max number of iterations reached.")
        else:
            for key, condition in stop_conditions.items():
                is_any_satisfied |= condition.is_satisfied(training_context)

                if(is_any_satisfied):
                    self.__logger.info('Condition named "{key}" is satisfied'.format(key=key))
                    break

        self.__logger.info("Finished checking stop conditions.")
        return is_any_satisfied

    async def train(self, model: TModel, dataset: Union[Tuple[str, Dataset[Tuple[TInput, TTarget]]], Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]], objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget]], primary_objective: Optional[str] = None, validation_dataset: Optional[Union[Tuple[str, Dataset[Tuple[TInput, TTarget]]], Dataset[Tuple[TInput, TTarget]]]] = None, logger: Optional[Logger] = None) -> TModel:
        logger = logger if not logger is None else self.__logger
        
        if model is None:
            raise ValueError("model")

        if dataset is None:
            raise ValueError("dataset")

        if stop_conditions is None:
            raise ValueError("stop_conditions")

        if objective_functions is None:
            raise ValueError("objective_functions can't be empty")

        if primary_objective is None:
            primary_objective = list(objective_functions.keys())[0]

        current_dataset: Tuple[str, Dataset[Tuple[TInput, TTarget]]] = None
        training_context: TrainingContext[TInput, TTarget, TModel] = None
        training_dataset: Tuple[str, Dataset[Tuple[TInput, TTarget]]] = None

        if isinstance(dataset, Tuple):
            current_dataset = dataset
        else:
            current_dataset = (type(dataset).__name__, dataset)

        training_context = TrainingContext[TInput, TTarget, TModel](model=model, dataset_name=current_dataset[0], scores={objective: deque([], self.__max_scores) for objective in objective_functions.keys()}, train_losses=deque([], self.__max_losses), _primary_objective=primary_objective, current_epoch=0, current_batch_index=0)

        training_size: int = int(len(current_dataset[1]) * self.__training_dataset_size_ratio)
        validation_size: int = int(len(current_dataset[1]) - training_size)

        if validation_dataset is None: 
            training_split, validation_split = random_split(current_dataset[1], [training_size, validation_size])

            training_dataset = (current_dataset[0], training_split)
            validation_dataset = (current_dataset[0], validation_split)
        else:
            training_dataset = current_dataset

        logger.info('Starting training loop...')

        self.__execute_pre_loop_plugins(logger, training_context)

        training_data_loader: DataLoader[Tuple[TInput, TTarget]] = DataLoader[Tuple[TInput, TTarget]](dataset=training_dataset[1], batch_size=self.__batch_size, drop_last=self.__drop_last)
        
        while not self.__is_any_stop_condition_satisfied(training_context=training_context, stop_conditions=stop_conditions):
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

            for batch_index, batch in enumerate(tqdm(training_data_loader, miniters=len(training_dataset[1])/100, initial=training_context.current_batch_index)):
                training_context.current_batch_index = batch_index

                iteration_start_time = time.time()

                sum_batch_load_time += iteration_start_time - batch_load_start_time
                count_batch_load_times += 1

                logger.debug(f"Batch load took {iteration_start_time - batch_load_start_time} seconds.")

                self.__execute_pre_train_plugins(logger, training_context)

                inputs: List[TInput] = [value[0] for value in batch]
                targets: List[TTarget] = [value[1] for value in batch]

                logger.debug("Executing training step.")
                train_loss: float = model.training_step(inputs, targets)
                
                training_context.train_losses.append(train_loss)

                self.__execute_post_train_plugins(logger, training_context)

                iteration_end_time = time.time()
                sum_iteration_run_time += iteration_end_time - iteration_start_time
                count_iteration_run_times += 1

                logger.debug(f"Iteration took {iteration_end_time - iteration_start_time} seconds.")

                batch_load_start_time = time.time()

            logger.info(f"Each batch load took around {sum_batch_load_time/count_batch_load_times} seconds.")
            logger.info(f"Each iteration took around {sum_iteration_run_time/count_iteration_run_times} seconds.")

            logger.info("Evaluating current model.")
            evaluation_scores: Dict[str, Score] = await self.__evaluation_service.evaluate(model=model, evaluation_dataset=validation_dataset, evaluation_metrics=objective_functions)
            logger.info("finished evaluating current model.")

            for key, evaluation_score in evaluation_scores.items():
                training_context.scores[key].append(evaluation_score)

            self.__execute_post_epoch_plugins(logger, training_context)

            logger.info("Finished epoch.")
            logger.info(f"Epoch took {time.time() - epoch_start_time} seconds.")

        logger.info("Finished training loop.")

        self.__execute_post_loop_plugins(logger, training_context)

        return model

    async def train_on_multiple_datasets(self, model: TModel, datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]], objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget]], primary_objective: Optional[str] = None, validation_dataset: Optional[Union[Tuple[str, Dataset[Tuple[TInput, TTarget]]], Dataset[Tuple[TInput, TTarget]]]] = None) -> TModel:
        self.__logger.info(f"Starting training on {len(datasets)} datasets...")

        context: MultiTrainingContext = MultiTrainingContext(current_dataset_index=0)

        self.__execute_pre_multi_loop_plugins(self.__logger, context)

        for dataset_index in range(context.current_dataset_index, len(datasets.items())):
            context.current_dataset_index = dataset_index

            self.__execute_pre_multi_train_step_plugins(self.__logger, context)

            dataset_name, dataset = list(datasets.items())[dataset_index]

            training_run_logger: Logger = self.__logger.getChild(dataset_name)
            model = await self.train(model, (dataset_name, dataset), stop_conditions, objective_functions, primary_objective, validation_dataset, training_run_logger)

            self.__execute_post_multi_train_step_plugins(self.__logger, context)

        self.__logger.info(f"Finished training on {len(datasets.items())} datasets.")

        self.__execute_post_multi_loop_plugins(self.__logger, context)

        return model