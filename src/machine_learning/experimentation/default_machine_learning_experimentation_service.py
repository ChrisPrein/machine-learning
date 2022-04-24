from ast import Call
import asyncio
import itertools
from logging import Logger
import logging
from tokenize import Single
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, Tuple, TypeVar, Union
from matplotlib.pyplot import eventplot
from torch.utils.data import Dataset, random_split
import nest_asyncio
from multipledispatch import dispatch

from .abstractions.instance_factory import InstanceFactory
from ..evaluation.abstractions.evaluation_metric import EvaluationMetric, EvaluationContext, TModel
from ..training.abstractions.stop_condition import StopCondition, TrainingContext
from ..parameter_tuning.abstractions.objective_function import ObjectiveFunction
from ..modeling.abstractions.model import TInput, TTarget
from ..evaluation.abstractions.evaluation_service import EvaluationService
from ..training.abstractions.training_service import TrainingService
from .abstractions.machine_learning_experimentation_service import MachineLearningExperimentationService, MachineLearningExperimentSettings, MachineLearningExperimentResult, MachineLearningRunResult, MachineLearningRunSettings, InstanceSettings
from .default_instance_factory import DefaultInstanceFactory
from .default_dict_instance_factory import DefaultDictInstanceFactory

nest_asyncio.apply()

TSettings = TypeVar('TSettings', InstanceSettings, Dict[str, InstanceSettings])
TInstance = TypeVar('TInstance')

FactoryAlias = Union[InstanceFactory[TSettings, TInstance], Callable[[TSettings], TInstance]]
ModelFactoryAlias = FactoryAlias[InstanceSettings, TModel]
TrainingServiceFactoryAlias = FactoryAlias[InstanceSettings, TrainingService[TInput, TTarget, TModel]]
EvaluationServiceFactoryAlias = FactoryAlias[InstanceSettings, EvaluationService[TInput, TTarget, TModel]]
TrainingDatasetFactoryAlias = FactoryAlias[Dict[str, InstanceSettings], Dict[str, Dataset[Tuple[TInput, TTarget]]]]
EvaluationDatasetFactoryAlias = FactoryAlias[Dict[str, InstanceSettings], Dict[str, Dataset[Tuple[TInput, TTarget]]]]
EvaluationMetricFactoryAlias = FactoryAlias[Dict[str, InstanceSettings], Dict[str, EvaluationMetric[TInput, TTarget, TModel]]]
ObjectiveFunctionFactoryAlias = FactoryAlias[Dict[str, InstanceSettings], Dict[str, ObjectiveFunction[TInput, TTarget, TModel]]]
StopConditionFactoryAlias = FactoryAlias[Dict[str, InstanceSettings], Dict[str, StopCondition[TModel]]]

class DefaultMachineLearningExperimentationService(MachineLearningExperimentationService[TModel]):
    def __init__(self, logger: Logger,
    model_factory: ModelFactoryAlias[TModel], 
    training_service_factory: TrainingServiceFactoryAlias[TInput, TTarget, TModel],
    evaluation_service_factory: EvaluationServiceFactoryAlias[TInput, TTarget, TModel],
    training_dataset_factory: TrainingDatasetFactoryAlias[TInput, TTarget], 
    test_dataset_factory: EvaluationDatasetFactoryAlias[TInput, TTarget],
    evaluation_metric_factory: EvaluationMetricFactoryAlias[TInput, TTarget, TModel],
    objective_function_factory: ObjectiveFunctionFactoryAlias[TInput, TTarget, TModel],
    stop_condition_factory: StopConditionFactoryAlias[TModel],
    event_loop: Optional[asyncio.AbstractEventLoop] = None):

        if logger is None:
            logger = logging.getLogger()

        if model_factory is None:
            raise ValueError("model_factory")

        if training_service_factory is None:
            raise ValueError("training_service_factory")

        if evaluation_service_factory is None:
            raise ValueError("evaluation_service_factory")

        if training_dataset_factory is None:
            raise ValueError("training_dataset_factory")

        if test_dataset_factory is None:
            raise ValueError("test_dataset_factory")        
            
        if evaluation_metric_factory is None:
            raise ValueError("evaluation_metric_factory")

        if objective_function_factory is None:
            raise ValueError("objective_function_factory")

        if stop_condition_factory is None:
            raise ValueError("stop_condition_factory")

        self.__logger: Logger = logger
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()
        self.__model_factory: ModelFactoryAlias[TModel] = model_factory
        self.__training_service_factory: TrainingServiceFactoryAlias[TInput, TTarget, TModel] = training_service_factory
        self.__evaluation_service_factory: EvaluationServiceFactoryAlias[TInput, TTarget, TModel] = evaluation_service_factory
        self.__training_dataset_factory: TrainingDatasetFactoryAlias[TInput, TTarget] = training_dataset_factory
        self.__test_dataset_factory: EvaluationDatasetFactoryAlias[TInput, TTarget] = test_dataset_factory
        self.__evaluation_metric_factory: EvaluationMetricFactoryAlias[TInput, TTarget, TModel] = evaluation_metric_factory
        self.__objective_function_factory: ObjectiveFunctionFactoryAlias[TInput, TTarget, TModel] = objective_function_factory
        self.__stop_condition_factory: StopConditionFactoryAlias[TModel] = stop_condition_factory

    async def __execute_run(self, run_settings: MachineLearningRunSettings) -> MachineLearningRunResult[TModel]:
        model: TModel = self.__model_factory(run_settings.model_settings)

        training_service: TrainingService[TInput, TTarget, TModel] = self.__training_service_factory(run_settings.training_service_settings)
        training_datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]] = self.__training_dataset_factory(run_settings.training_dataset_settings)
        stop_conditions: Dict[str, StopCondition[TModel]] =  self.__stop_condition_factory(run_settings.stop_condition_settings)
        objective_functions: Dict[str, ObjectiveFunction[TInput, TTarget, TModel]] = self.__objective_function_factory(run_settings.objective_function_settings)

        model = await training_service.train_on_multiple_datasets(model=model, datasets=training_datasets, stop_conditions=stop_conditions, objective_functions=objective_functions)

        evaluation_service: EvaluationService[TInput, TTarget, TModel] = self.__evaluation_service_factory(run_settings.evaluation_service_settings)
        evaluation_datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]] = self.__test_dataset_factory(run_settings.evaluation_dataset_settings)
        evaluation_metrics: Dict[str, EvaluationMetric[TInput, TTarget, TModel]] = self.__evaluation_metric_factory(run_settings.evaluation_metric_settings)
        
        scores: Dict[str, Dict[str, float]] = await evaluation_service.evaluate_on_multiple_datasets(model=model, evaluation_datasets=evaluation_datasets, evaluation_metrics=evaluation_metrics)

        return MachineLearningRunResult[TModel](run_settings=run_settings, model=model, scores=scores)

    async def run_experiment(self, experiment_settings: MachineLearningExperimentSettings) -> MachineLearningExperimentResult[TModel]:
        combinations: List[Tuple] = itertools.product(experiment_settings.model_settings, experiment_settings.training_service_settings, 
        experiment_settings.evaluation_service_settings, experiment_settings.training_dataset_settings, experiment_settings.evaluation_dataset_settings, 
        experiment_settings.evaluation_metric_settings, experiment_settings.objective_function_settings, experiment_settings.stop_condition_settings)

        runs: List[MachineLearningRunSettings] = [MachineLearningRunSettings(*combination) for combination in combinations]

        run_tasks: List[Coroutine[Any, Any, MachineLearningRunResult[TModel]]] = [self.__execute_run(run_settings) for run_settings in runs]

        completed, pending = await asyncio.wait(run_tasks)

        return MachineLearningExperimentResult[TModel]([t.result() for t in completed])

    async def __run_experiment(self, experiment_settings: Tuple[str, MachineLearningExperimentSettings]) -> Tuple[str, MachineLearningExperimentResult[TModel]]:
        result = await self.run_experiment(experiment_settings[1])

        return (experiment_settings[0], result)

    async def run_experiments(self, experiment_settings: Dict[str, MachineLearningExperimentSettings]) -> Dict[str, MachineLearningExperimentResult[TModel]]:
        experiment_tasks: List[Coroutine[Any, Any, Tuple[str, MachineLearningExperimentResult[TModel]]]] = [self.__run_experiment(settings) for settings in experiment_settings.items()]

        completed, pending = await asyncio.wait(experiment_tasks)

        return dict([t.result() for t in completed])