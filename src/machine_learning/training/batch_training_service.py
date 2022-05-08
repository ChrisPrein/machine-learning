from abc import ABC
from logging import Logger
import logging
from typing import Any, Coroutine, TypeVar, List, Generic, Optional, Dict, Tuple, Union
import asyncio
import asyncio.tasks
import asyncio.futures
from uuid import UUID
import uuid
from dataset_handling.dataloader import DataLoader
import torch
from torch.utils.data import Dataset, random_split
from multidispatch import multimethod, multifunction
import nest_asyncio
import zope.event

from machine_learning.modeling.pytorch_model import PytorchModel

from ..evaluation.abstractions.evaluation_service import EvaluationService
from ..parameter_tuning.abstractions.objective_function import ObjectiveFunction
from .abstractions.stop_condition import StopCondition, TrainingContext, Score
from ..modeling.abstractions.model import Model, TInput, TTarget
from .abstractions.training_service import TrainingService
from ..evaluation.abstractions.evaluation_metric import EvaluationContext, TModel
from..evaluation.multi_task_evaluation_service import MultiTaskEvaluationService

CHECKING_STOP_CONDITIONS = 70
FINISHED_CHECKING_STOP_CONDITIONS = 71
MAX_NUMBER_EPOCHS_REACHED = 72
MAX_NUMBER_ITERATIONS_REACHED = 73
STOP_CONDITION_SATISFIED = 74
STARTING_TRAINING = 75
FINISHED_TRAINING = 76
STARTING_MULTI_DATASET_TRAINING = 77
FINISHED_MULTI_DATASET_TRAINING = 78

TRAINING_LOGGER_NAME = "training"

nest_asyncio.apply()

class BatchTrainingService(TrainingService[TInput, TTarget, TModel], ABC):
    def __init__(self, logger: Optional[Logger]=None, evaluation_service: Optional[EvaluationService[TInput, TTarget, TModel]] = None, batch_size: Optional[int] = None, drop_last: bool = True, event_loop: Optional[asyncio.AbstractEventLoop] = None, max_epochs: int = 100, max_iterations: int = 10000, training_dataset_size_ratio: float = 0.8):
        if logger is None:
            self.__logger: Logger = logging.getLogger()
        else:
            self.__logger: Logger = logger.getChild(TRAINING_LOGGER_NAME)

        if evaluation_service is None:
            evaluation_service = MultiTaskEvaluationService[TInput, TTarget, TModel](logger=self.__logger, batch_size=batch_size, drop_last=drop_last, event_loop=event_loop)
        
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()
        self.__max_epochs: int = max_epochs
        self.__max_iterations: int = max_iterations
        self.__batch_size: Optional[int] = batch_size
        self.__drop_last: bool = drop_last
        self.__evaluation_service: EvaluationService[TInput, TTarget, TModel, EvaluationContext[TInput, TTarget, TModel]] = evaluation_service
        self.__training_dataset_size_ratio: float = training_dataset_size_ratio

    def is_any_stop_condition_satisfied(self, training_context: TrainingContext[TModel], stop_conditions: Dict[str, StopCondition[TModel]]) -> bool:
        self.__logger.log(CHECKING_STOP_CONDITIONS, {"training_context": training_context, "stop_conditions": stop_conditions})
        self.__logger.info("Checking stop conditions...")
        is_any_satisfied: bool = False

        if training_context.current_epoch > self.__max_epochs: 
            self.__logger.log(MAX_NUMBER_EPOCHS_REACHED, training_context)
            self.__logger.info("Max number of epochs reached.")
            is_any_satisfied = True
        elif training_context.current_iteration > self.__max_iterations:
            self.__logger.log(MAX_NUMBER_ITERATIONS_REACHED, training_context)
            self.__logger.info("Max number of iterations reached.")
        else:
            for key, condition in stop_conditions.items():
                is_any_satisfied |= condition.is_satisfied(training_context)

                if(is_any_satisfied):
                    self.__logger.log(STOP_CONDITION_SATISFIED, {"training_context": training_context, "stop_condition_name": key, "stop_condition": condition})
                    self.__logger.info('Condition named "{key}" is satisfied'.format(key=key))
                    break

        self.__logger.log(FINISHED_CHECKING_STOP_CONDITIONS, {"training_context": training_context, "stop_conditions": stop_conditions})
        self.__logger.info("Finished checking stop conditions.")
        return is_any_satisfied

    async def train(self, model: TModel, dataset: Union[Tuple[str, Dataset[Tuple[TInput, TTarget]]], Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TModel]], objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget, TModel]], primary_objective: Optional[str] = None, validation_dataset: Optional[Dataset[Tuple[TInput, TTarget]]] = None, logger: Optional[Logger] = None) -> TModel:
        training_run_id: UUID = uuid.uuid4()

        if logger is None:
            training_run_logger: Logger = self.__logger.getChild(str(training_run_id))
        else:
            training_run_logger: Logger = logger.getChild(str(training_run_id))
        
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

        batch_size: int = len(dataset) if self.__batch_size is None else self.__batch_size
        training_context: TrainingContext[TModel] = TrainingContext[TModel](model=model, scores={objective: [] for objective in objective_functions.keys()}, _primary_objective=primary_objective, current_epoch=0, current_iteration=0)

        current_dataset: Dataset[Tuple[TInput, TTarget]] = None

        if isinstance(dataset, Tuple):
            current_dataset = dataset[1]
        else:
            current_dataset = dataset

        training_size: int = int(len(current_dataset) * self.__training_dataset_size_ratio)
        validation_size: int = int(len(current_dataset) - training_size)

        if validation_dataset is None: 
            training_dataset, validation_dataset = random_split(current_dataset, [training_size, validation_size])
        else:
            training_dataset = current_dataset

        training_data_loader: DataLoader[Tuple[TInput, TTarget]] = DataLoader[Tuple[TInput, TTarget]](dataset=training_dataset, batch_size=batch_size, drop_last=self.__drop_last)
        training_run_logger.log(STARTING_TRAINING, {"model": model, "dataset": dataset, "stop_conditions": stop_conditions, "objective_functions": objective_functions, "primary_objective": primary_objective, "batch_size": batch_size})
        training_run_logger.info('Starting training loop...')

        if isinstance(model, PytorchModel) and not model.scheduler_factory is None:
            model.scheduler = model.scheduler_factory()

        while not self.is_any_stop_condition_satisfied(training_context=training_context, stop_conditions=stop_conditions):
            training_context.current_epoch += 1

            for batch_index, batch in enumerate(training_data_loader):
                training_context.current_iteration += 1
                inputs: List[TInput] = [value[0] for value in batch]
                targets: List[TTarget] = [value[1] for value in batch]
                model.train_batch(input_batch=inputs, target_batch=targets)

            training_run_logger.info("Evaluating current model.")
            evaluation_scores: Dict[str, float] = await self.__evaluation_service.evaluate(model=model, evaluation_dataset=validation_dataset, evaluation_metrics=objective_functions)
            training_run_logger.info("finished evaluating current model.")

            if isinstance(model, PytorchModel) and not model.scheduler is None:
                model.scheduler.step()

            for key, evaluation_score in evaluation_scores.items():
                training_context.scores[key].append(Score(epoch=training_context.current_epoch, iteration=training_context.current_iteration, score=evaluation_score, optimization_type=objective_functions[key].optimization_type))

        training_run_logger.log(FINISHED_TRAINING, {"model": model, "dataset": dataset, "stop_conditions": stop_conditions, "objective_functions": objective_functions, "primary_objective": primary_objective, "batch_size": batch_size})
        training_run_logger.info("Finished training loop.")

        return model

    async def __train(self, model: TModel, dataset: Tuple[str, Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TModel]], objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget, TModel]], logger: Logger, primary_objective: Optional[str] = None, validation_dataset: Optional[Dataset[Tuple[TInput, TTarget]]] = None) -> TModel:
        result = await self.train(model, dataset, stop_conditions, objective_functions, primary_objective, validation_dataset, logger)

        return result

    async def train_on_multiple_datasets(self, model: TModel, datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TModel]], objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget, TModel]], primary_objective: Optional[str] = None, validation_dataset: Optional[Dataset[Tuple[TInput, TTarget]]] = None) -> TModel:
        multi_training_run_id: UUID = uuid.uuid4()

        multi_training_run_logger: Logger = self.__logger.getChild(str(multi_training_run_id))

        multi_training_run_logger.log(STARTING_MULTI_DATASET_TRAINING, {"model": model, "dataset": datasets, "stop_conditions": stop_conditions, "objective_functions": objective_functions, "primary_objective": primary_objective, "batch_size": self.__batch_size})
        multi_training_run_logger.info(f"starting training on {len(datasets)} datasets...")
        
        for dataset in datasets.items():
            model = await self.__train(model, dataset, stop_conditions, objective_functions, multi_training_run_logger, primary_objective, validation_dataset)

        multi_training_run_logger.log(FINISHED_MULTI_DATASET_TRAINING, {"model": model, "dataset": datasets, "stop_conditions": stop_conditions, "objective_functions": objective_functions, "primary_objective": primary_objective, "batch_size": self.__batch_size})
        multi_training_run_logger.info(f"finished training on {len(datasets)} datasets.")

        return model