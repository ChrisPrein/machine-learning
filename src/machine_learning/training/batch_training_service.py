from abc import ABC
from logging import Logger
from typing import Any, Coroutine, TypeVar, List, Generic, Optional, Dict, Tuple
import asyncio
import asyncio.tasks
import asyncio.futures
from dataset_handling.dataloader import DataLoader
from torch.utils.data import Dataset, random_split
from multidispatch import multimethod, multifunction
import nest_asyncio

from ..evaluation.abstractions.evaluation_service import EvaluationService
from ..parameter_tuning.abstractions.objective_function import ObjectiveFunction
from .abstractions.stop_condition import StopCondition, TrainingContext, Score
from ..modeling.abstractions.model import Model, TInput, TTarget
from .abstractions.training_service import TrainingService
from ..evaluation.abstractions.evaluation_metric import EvaluationContext, TModel

nest_asyncio.apply()

class BatchTrainingService(TrainingService[TInput, TTarget, TModel], ABC):
    def __init__(self, logger: Logger, evaluation_service: EvaluationService[TInput, TTarget, TModel], batch_size: Optional[int], drop_last: bool = True, event_loop: Optional[asyncio.AbstractEventLoop] = None, max_epochs: int = 100, max_iterations: int = 10000, training_dataset_size_ratio: float = 0.8):
        if logger is None:
            raise ValueError("logger")

        if evaluation_service is None:
            raise ValueError("evaluation_service")
        
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()
        self.__max_epochs: int = max_epochs
        self.__max_iterations: int = max_iterations
        self.__batch_size: Optional[int] = batch_size
        self.__drop_last: bool = drop_last
        self.__logger: Logger = Logger
        self.__evaluation_service: EvaluationService[TInput, TTarget, TModel, EvaluationContext[TInput, TTarget, TModel]] = evaluation_service
        self.__training_dataset_size_ratio: float = training_dataset_size_ratio

    def is_any_stop_condition_satisfied(self, training_context: TrainingContext[TModel], stop_conditions: Dict[str, StopCondition[TModel]]) -> bool:
        self.__logger.info("Checking stop conditions.")
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
                    self.__logger('Condition named "{key}" is satisfied'.format(key=key))
                    break

        self.__logger.info("Finished checking stop conditions.")
        return is_any_satisfied

    @multimethod(Model, Dataset, dict, dict, str)
    async def train(self, model: TModel, dataset: Dataset[Tuple[TInput, TTarget]], stop_conditions: Dict[str, StopCondition[TModel]], objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget, TModel]], primary_objective: Optional[str] = None) -> TModel:
        if model is None:
            raise ValueError("model")

        if dataset is None:
            raise ValueError("dataset")

        if stop_conditions is None:
            raise ValueError("stop_conditions")

        if objective_functions is None:
            raise ValueError("objective_functions can't be empty")

        if primary_objective is None:
            primary_objective = objective_functions.keys()[0]

        batch_size: int = len(dataset) if self.__batch_size is None else self.__batch_size
        training_context: TrainingContext[TModel] = TrainingContext[TModel](model=model, scores={objective: [] for objective in objective_functions.keys()}, _primary_objective=primary_objective, scores=[], current_epoch=0, current_iteration=0)
        training_size: int = len(dataset) * self.__training_dataset_size_ratio
        validation_size: int = 1 - training_size
        training_dataset, validation_dataset = random_split(dataset=dataset, lengths=(training_size, validation_size))
        training_data_loader: DataLoader[Tuple[TInput, TTarget]] = DataLoader[Tuple[TInput, TTarget]](dataset=training_dataset, batch_size=batch_size, drop_last=self.__drop_last)
        self.__logger.info("Starting training loop.")

        while not self.is_any_stop_condition_satisfied(training_context=training_context, stop_conditions=stop_conditions):
            training_context.current_epoch += 1

            for batch_index, batch in enumerate(training_data_loader):
                training_context.current_iteration += 1
                inputs: List[TInput] = [value[0] for value in batch]
                targets: List[TTarget] = [value[1] for value in batch]
                model.train_batch(input_batch=inputs, target_batch=targets)

            self.__logger.info("Evaluating current model.")
            evaluation_scores: Dict[str, float] = await self.__evaluation_service.evaluate(model=model, evaluation_dataset=validation_dataset, evaluation_metrics=objective_functions)
            self.__logger.info("finished evaluating current model.")

            for key, evaluation_score in evaluation_scores:
                training_context.scores[key].append(Score(epoch=training_context.current_epoch, iteration=training_context.current_iteration, score=evaluation_score, optimization_type=objective_functions[key].optimization_type))

        self.__logger.info("Finished training loop.")

        return model

    async def __train(self, model: TModel, dataset: Tuple[str, Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TModel]], objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget, TModel]], primary_objective: Optional[str] = None) -> TModel:
        result = await self.train(model, dataset[1], stop_conditions, objective_functions, primary_objective)

        return result

    @train.dispatch(Model, dict, dict, dict, str)
    async def train(self, model: TModel, datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]], stop_conditions: Dict[str, StopCondition[TModel]], objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget, TModel]], primary_objective: Optional[str] = None) -> TModel:
        for dataset in datasets:
            model = await self.__train(model, dataset, stop_conditions, objective_functions, primary_objective)

        return model