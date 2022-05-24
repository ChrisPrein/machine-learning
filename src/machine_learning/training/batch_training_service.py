from abc import ABC
from logging import Logger
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine, TypeVar, List, Generic, Optional, Dict, Tuple, Union
import asyncio
import asyncio.tasks
import asyncio.futures
from uuid import UUID
import uuid
from dataset_handling.dataloader import DataLoader
from torch.utils.data import Dataset, random_split
import nest_asyncio
from dataclasses import dataclass

from ..evaluation.default_evaluation import default_evaluation
from ..evaluation.abstractions.evaluation_service import EvaluationService
from ..parameter_tuning.abstractions.objective_function import ObjectiveFunction
from .abstractions.stop_condition import StopCondition, TrainingContext, Score
from ..modeling.abstractions.model import Model, TInput, TTarget
from .abstractions.training_service import TrainingService
from ..evaluation.abstractions.evaluation_metric import EvaluationContext, TModel
from ..evaluation.multi_task_evaluation_service import MultiTaskEvaluationService

nest_asyncio.apply()

@dataclass
class TrainingCheckpoint(Generic[TInput, TTarget, TModel]):
    training_dataset: Tuple[str, Dataset[Tuple[TInput, TTarget]]]
    validation_dataset: Tuple[str, Dataset[Tuple[TInput, TTarget]]]
    training_context: TrainingContext[TInput, TTarget, TModel]
    stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]]
    objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget, TModel]]
    primary_objective: str

class BatchTrainingService(TrainingService[TInput, TTarget, TModel], ABC):
    def __init__(self, train_hook: Callable[[Logger, TrainingContext[TInput, TTarget, TModel], List[TInput], List[TTarget]], None], 
    evaluation_hook: Callable[[Logger, TModel, List[TInput], List[TTarget]], List[TTarget]] = default_evaluation, logger: Optional[Logger]=None, 
    evaluation_service: Optional[EvaluationService[TInput, TTarget, TModel]] = None, 
    batch_size: int = 1, drop_last: bool = True, event_loop: Optional[asyncio.AbstractEventLoop] = None, max_epochs: int = 100, 
    max_iterations: int = 10000, training_dataset_size_ratio: float = 0.8, pre_loop_hook: Optional[Callable[[Logger, TrainingContext[TInput, TTarget, TModel]], None]] = None,
    post_loop_hook: Optional[Callable[[Logger, TrainingContext[TInput, TTarget, TModel]], None]] = None, pre_epoch_hook: Optional[Callable[[Logger, TrainingContext[TInput, TTarget, TModel]], None]] = None, 
    post_epoch_hook: Optional[Callable[[Logger, TrainingContext[TInput, TTarget, TModel], Dataset[Tuple[TInput, TTarget]]], None]] = None, pre_train_hook: Optional[Callable[[Logger, TrainingContext[TInput, TTarget, TModel]], None]] = None,
    post_train_hook: Optional[Callable[[Logger, TrainingContext[TInput, TTarget, TModel]], None]] = None, save_checkpoint_hook: Optional[Callable[[Logger, TrainingCheckpoint[TInput, TTarget, TModel]], None]] = None,
    load_checkpoint_hook: Optional[Callable[[Logger], Optional[TrainingCheckpoint[TInput, TTarget, TModel]]]] = None):
        
        if train_hook is None:
            raise ValueError("train_hook")

        self.__logger = logger if not logger is None else logging.getLogger()
        
        if evaluation_service is None:
            evaluation_service = MultiTaskEvaluationService[TInput, TTarget, TModel](logger=self.__logger, batch_size=batch_size, drop_last=drop_last, event_loop=event_loop, evaluation_hook=evaluation_hook)
        
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()
        self.__max_epochs: int = max_epochs
        self.__max_iterations: int = max_iterations
        self.__batch_size: int = batch_size
        self.__drop_last: bool = drop_last
        self.__evaluation_service: EvaluationService[TInput, TTarget, TModel, EvaluationContext[TInput, TTarget, TModel]] = evaluation_service
        self.__training_dataset_size_ratio: float = training_dataset_size_ratio

        self.__train_hook: Callable[[Logger, TrainingContext[TInput, TTarget, TModel], List[TInput], List[TTarget]], None] = train_hook
        self.__pre_loop_hook: Optional[Callable[[Logger, TrainingContext[TInput, TTarget, TModel]], None]] = pre_loop_hook
        self.__post_loop_hook: Optional[Callable[[Logger, TrainingContext[TInput, TTarget, TModel]], None]] = post_loop_hook
        self.__pre_epoch_hook: Optional[Callable[[Logger, TrainingContext[TInput, TTarget, TModel]], None]] = pre_epoch_hook
        self.__post_epoch_hook: Optional[Callable[[Logger, TrainingContext[TInput, TTarget, TModel], Dataset[Tuple[TInput, TTarget]]], None]] = post_epoch_hook
        self.__pre_train_hook: Optional[Callable[[Logger, TrainingContext[TInput, TTarget, TModel]], None]] = pre_train_hook
        self.__post_train_hook: Optional[Callable[[Logger, TrainingContext[TInput, TTarget, TModel]], None]] = post_train_hook

        self.__save_checkpoint_hook: Optional[Callable[[Logger, TrainingCheckpoint[TInput, TTarget, TModel]], None]] = save_checkpoint_hook
        self.__load_checkpoint_hook: Optional[Callable[[Logger], Optional[TrainingCheckpoint[TInput, TTarget, TModel]]]] = load_checkpoint_hook

    def is_any_stop_condition_satisfied(self, training_context: TrainingContext[TInput, TTarget, TModel], stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]], logger: Optional[Logger] = None) -> bool:
        if logger is None:
            logger = self.__logger
        
        self.__logger.info("Checking stop conditions...")
        is_any_satisfied: bool = False

        if training_context.current_epoch > self.__max_epochs: 
            self.__logger.info("Max number of epochs reached.")
            is_any_satisfied = True
        elif training_context.current_iteration > self.__max_iterations:
            self.__logger.info("Max number of iterations reached.")
        else:
            for key, condition in stop_conditions.items():
                is_any_satisfied |= condition.is_satisfied(training_context)

                if(is_any_satisfied):
                    self.__logger.info('Condition named "{key}" is satisfied'.format(key=key))
                    break

        self.__logger.info("Finished checking stop conditions.")
        return is_any_satisfied

    async def train(self, model: TModel, dataset: Union[Tuple[str, Dataset[Tuple[TInput, TTarget]]], Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]], objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget, TModel]], primary_objective: Optional[str] = None, validation_dataset: Optional[Dataset[Tuple[TInput, TTarget]]] = None, logger: Optional[Logger] = None) -> TModel:
        if logger is None:
            logger = self.__logger
        
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
        checkpoint: Optional[TrainingCheckpoint[TInput, TTarget, TModel]] = None
        training_dataset: Tuple[Dataset[Tuple[TInput, TTarget]]] = None
        validation_dataset: Tuple[Dataset[Tuple[TInput, TTarget]]] = None
        training_data_loader: DataLoader[Tuple[TInput, TTarget]] = None

        if not self.__load_checkpoint_hook is None:
            logger.info("Loading last checkpoint...")
            checkpoint = self.__load_checkpoint_hook(logger)

        if checkpoint is None:
            if isinstance(dataset, Tuple):
                current_dataset = dataset
            else:
                current_dataset = (type(dataset).__name__, dataset)

            training_context = TrainingContext[TInput, TTarget, TModel](model=model, dataset_name=current_dataset[0], scores={objective: [] for objective in objective_functions.keys()}, _primary_objective=primary_objective, current_epoch=0, current_iteration=0)

            training_size: int = int(len(current_dataset) * self.__training_dataset_size_ratio)
            validation_size: int = int(len(current_dataset) - training_size)

            if validation_dataset is None: 
                training_split, validation_split = random_split(current_dataset, [training_size, validation_size])

                training_dataset = (current_dataset[0], training_split)
                validation_dataset = (current_dataset[0], validation_split)
            else:
                training_dataset = current_dataset

            training_data_loader = DataLoader[Tuple[TInput, TTarget]](dataset=training_dataset[1], batch_size=self.__batch_size, drop_last=self.__drop_last)
            logger.info('Starting training loop...')

            if not self.__pre_loop_hook is None:
                logger.debug("Executing pre loop hook.")
                self.__pre_loop_hook(logger, training_context)
        else:
            training_context = checkpoint.training_context
            model = checkpoint.training_context.model
            training_dataset = checkpoint.training_dataset
            validation_dataset = checkpoint.validation_dataset

            training_data_loader = DataLoader[Tuple[TInput, TTarget]](dataset=training_dataset[1], batch_size=self.__batch_size, drop_last=self.__drop_last)

            logger.info("Continuing training loop from last checkpoint...")

        while not self.is_any_stop_condition_satisfied(training_context=training_context, stop_conditions=stop_conditions):
            training_context.current_epoch += 1

            if not self.__pre_epoch_hook is None:
                logger.debug("Executing pre epoch hook.")
                self.__pre_train_hook(logger, training_context)

            for batch_index, batch in enumerate(training_data_loader):
                training_context.current_iteration += 1

                if not self.__pre_train_hook is None:
                    logger.debug("Executing pre training hook.")
                    self.__pre_train_hook(logger, training_context)

                inputs: List[TInput] = [value[0] for value in batch]
                targets: List[TTarget] = [value[1] for value in batch]

                logger.debug("Executing training hook.")
                self.__train_hook(logger, training_context, inputs, targets)

                if not self.__post_train_hook is None:
                    logger.debug("Executing post training hook.")
                    self.__post_train_hook(logger, training_context)

            logger.info("Evaluating current model.")
            evaluation_scores: Dict[str, Score] = await self.__evaluation_service.evaluate(model=model, evaluation_dataset=validation_dataset, evaluation_metrics=objective_functions)
            logger.info("finished evaluating current model.")

            for key, evaluation_score in evaluation_scores.items():
                training_context.scores[key].append(evaluation_score)

            if not self.__post_epoch_hook is None:
                logger.debug("Executing post epoch hook.")
                self.__post_epoch_hook(logger, training_context[1], validation_dataset[1])

            logger.info({'training_context': training_context})

            if not self.__save_checkpoint_hook is None:
                logger.info("creating checkpoint...")

                new_checkpoint: TrainingCheckpoint[TInput, TTarget, TModel] = TrainingCheckpoint[TInput, TTarget, TModel](training_dataset=training_dataset, validation_dataset=validation_dataset,
                training_context=training_context, stop_conditions=stop_conditions, objective_functions=objective_functions, primary_objective=primary_objective)

                self.__save_checkpoint_hook(logger, new_checkpoint)

                logger.info("checkpoint created.")

        
        logger.info("Finished training loop.")

        if not self.__post_loop_hook is None:
            logger.debug("Executing post loop hook.")
            self.__post_loop_hook(logger, training_context)

        logger.info({'model': model})

        return model

    async def __train(self, model: TModel, dataset: Tuple[str, Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]], objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget, TModel]], logger: Logger, primary_objective: Optional[str] = None, validation_dataset: Optional[Dataset[Tuple[TInput, TTarget]]] = None) -> TModel:
        result = await self.train(model, dataset, stop_conditions, objective_functions, primary_objective, validation_dataset, logger)

        return result

    async def train_on_multiple_datasets(self, model: TModel, datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TInput, TTarget, TModel]], objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget, TModel]], primary_objective: Optional[str] = None, validation_dataset: Optional[Dataset[Tuple[TInput, TTarget]]] = None) -> TModel:
        self.__logger.info(f"starting training on {len(datasets)} datasets...")
        
        for dataset in datasets.items():
            training_run_logger: Logger = self.__logger.getChild(str(dataset[0]))
            model = await self.__train(model, dataset, stop_conditions, objective_functions, training_run_logger, primary_objective, validation_dataset)

        self.__logger.info(f"finished training on {len(datasets)} datasets.")

        return model