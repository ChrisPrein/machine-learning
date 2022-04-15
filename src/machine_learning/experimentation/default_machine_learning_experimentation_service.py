from ast import Call
import asyncio
from logging import Logger
from tokenize import Single
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, Tuple, Union
from experimentation.experiment.abstractions.experimentation_service import ExperimentationService, TExperimentSettings, TExperimentResult
from matplotlib.pyplot import eventplot
from torch.utils.data import Dataset, random_split
import nest_asyncio
from multipledispatch import dispatch

from .abstractions.multi_instance_factory import MultiInstanceFactory
from .abstractions.single_instance_factory import SingleInstanceFactory
from ..evaluation.abstractions.evaluation_metric import EvaluationMetric
from ..training.abstractions.stop_condition import StopCondition
from ..parameter_tuning.abstractions.objective_function import ObjectiveFunction
from ..modeling.abstractions.model import TInput, TTarget
from ..evaluation.abstractions.evaluation_context import TModel, EvaluationContext
from ..training.abstractions.training_context import TrainingContext
from ..evaluation.abstractions.evaluation_service import EvaluationService, TEvaluationContext
from ..training.abstractions.training_service import TTrainingContext, TrainingService
from .default_machine_learning_experiment_result import DefaultMachineLearningExperimentResult
from .default_machine_learning_experiment_settings import DefaultMachineLearningExperimentSettings, TTrainingServiceSettings, TStopConditionSettings, TObjectiveFunctionSettings, TModelSettings, TEvaluationServiceSettings, TEvaluationMetricSettings, TTrainingDatasetSettings, TEvaluationDatasetSettings

nest_asyncio.apply()

class DefaultMachineLearningExperimentationService(Generic[TInput, TTarget, TModel, TTrainingServiceSettings, TStopConditionSettings, TObjectiveFunctionSettings, 
TModelSettings, TEvaluationServiceSettings, TEvaluationMetricSettings, TTrainingDatasetSettings, TEvaluationDatasetSettings], 
ExperimentationService[DefaultMachineLearningExperimentSettings, DefaultMachineLearningExperimentResult[TModel]]):
    def __init__(self, logger: Logger,
    model_factory: Union[SingleInstanceFactory[TModelSettings, TModel], Callable[[TModelSettings], TModel]], 
    training_service_factory: Union[SingleInstanceFactory[TTrainingServiceSettings, TrainingService[TInput, TTarget, TModel, TrainingContext[TModel], EvaluationContext[TInput, TTarget, TModel]]], Callable[[TTrainingServiceSettings], TrainingService[TInput, TTarget, TModel, TrainingContext[TModel], EvaluationContext[TInput, TTarget, TModel]]]], 
    evaluation_service_factory: Union[SingleInstanceFactory[TEvaluationServiceSettings,  EvaluationService[TInput, TTarget, TModel, EvaluationContext[TInput, TTarget, TModel]]], Callable[[TEvaluationServiceSettings], EvaluationService[TInput, TTarget, TModel, EvaluationContext[TInput, TTarget, TModel]]]], 
    training_dataset_factory: Union[MultiInstanceFactory[TTrainingDatasetSettings, Dataset[Tuple[TInput, TTarget]]], Callable[[TTrainingDatasetSettings], Dict[str, Dataset[Tuple[TInput, TTarget]]]]], 
    test_dataset_factory: Union[MultiInstanceFactory[TEvaluationDatasetSettings, Dataset[Tuple[TInput, TTarget]]], Callable[[TEvaluationDatasetSettings], Dict[str, Dataset[Tuple[TInput, TTarget]]]]],
    evaluation_metric_factory: Union[MultiInstanceFactory[TEvaluationMetricSettings, EvaluationMetric[EvaluationContext[TInput, TTarget, TModel]]], Callable[[TEvaluationMetricSettings], Dict[str, EvaluationMetric[EvaluationContext[TInput, TTarget, TModel]]]]],
    objective_function_factory: Union[MultiInstanceFactory[TObjectiveFunctionSettings, ObjectiveFunction[EvaluationContext[TInput, TTarget, TModel]]], Callable[[TObjectiveFunctionSettings], Dict[str, ObjectiveFunction[EvaluationContext[TInput, TTarget, TModel]]]]],
    stop_condition_factory: Union[MultiInstanceFactory[TStopConditionSettings, StopCondition[TrainingContext[TModel]]], Callable[[TStopConditionSettings], Dict[str, StopCondition[TrainingContext[TModel]]]]],
    event_loop: Optional[asyncio.AbstractEventLoop] = None):

        if logger is None:
            raise ValueError("logger")

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

        self.__logger: Logger = Logger
        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()
        self.__model_factory: Union[SingleInstanceFactory[TModelSettings, TModel], Callable[[TModelSettings], TModel]] = model_factory
        self.__training_service_factory: Union[SingleInstanceFactory[TTrainingServiceSettings, TrainingService[TInput, TTarget, TModel, TrainingContext[TModel], EvaluationContext[TInput, TTarget, TModel]]], Callable[[TTrainingServiceSettings], TrainingService[TInput, TTarget, TModel, TrainingContext[TModel], EvaluationContext[TInput, TTarget, TModel]]]] = training_service_factory
        self.__evaluation_service_factory: Union[SingleInstanceFactory[TEvaluationServiceSettings,  EvaluationService[TInput, TTarget, TModel, EvaluationContext[TInput, TTarget, TModel]]], Callable[[TEvaluationServiceSettings], EvaluationService[TInput, TTarget, TModel, EvaluationContext[TInput, TTarget, TModel]]]] = evaluation_service_factory
        self.__training_dataset_factory: Union[MultiInstanceFactory[TTrainingDatasetSettings, Dataset[Tuple[TInput, TTarget]]], Callable[[TTrainingDatasetSettings], Dict[str, Dataset[Tuple[TInput, TTarget]]]]] = training_dataset_factory
        self.__test_dataset_factory: Union[MultiInstanceFactory[TEvaluationDatasetSettings, Dataset[Tuple[TInput, TTarget]]], Callable[[TEvaluationDatasetSettings], Dict[str, Dataset[Tuple[TInput, TTarget]]]]] = test_dataset_factory
        self.__evaluation_metric_factory: Union[MultiInstanceFactory[TEvaluationMetricSettings, EvaluationMetric[EvaluationContext[TInput, TTarget, TModel]]], Callable[[TEvaluationMetricSettings], Dict[str, EvaluationMetric[EvaluationContext[TInput, TTarget, TModel]]]]] = test_dataset_factory
        self.__objective_function_factory: Union[MultiInstanceFactory[TObjectiveFunctionSettings, ObjectiveFunction[EvaluationContext[TInput, TTarget, TModel]]], Callable[[TObjectiveFunctionSettings], Dict[str, ObjectiveFunction[EvaluationContext[TInput, TTarget, TModel]]]]] = test_dataset_factory
        self.__stop_condition_factory: Union[MultiInstanceFactory[TStopConditionSettings, StopCondition[TrainingContext[TModel]]], Callable[[TStopConditionSettings], Dict[str, StopCondition[TrainingContext[TModel]]]]] = test_dataset_factory

    async def run_experiment(self, experiment_settings: DefaultMachineLearningExperimentSettings) -> DefaultMachineLearningExperimentResult[TModel]:
        model: TModel = self.__model_factory.create(experiment_settings.model_settings) if isinstance(self.__model_factory, SingleInstanceFactory[TModelSettings, TModel]) else self.__model_factory(experiment_settings.model_settings)
        
        training_service: TrainingService[TInput, TTarget, TModel, TrainingContext[TModel], EvaluationContext[TInput, TTarget, TModel]] = self.__training_service_factory.create(experiment_settings.training_service_settings) if isinstance(self.__training_service_factory, SingleInstanceFactory[TTrainingServiceSettings, TrainingService[TInput, TTarget, TModel, TrainingContext[TModel], EvaluationContext[TInput, TTarget, TModel]]]) else self.__training_service_factory(experiment_settings.training_service_settings)
        training_datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]] = self.__training_dataset_factory.create(experiment_settings.training_dataset_settings) if isinstance(self.__training_dataset_factory, MultiInstanceFactory[TTrainingDatasetSettings, Dataset[Tuple[TInput, TTarget]]]) else self.__training_dataset_factory(experiment_settings.training_dataset_settings)
        stop_conditions: Dict[str, StopCondition[TrainingContext[TModel]]] = self.__stop_condition_factory.create(experiment_settings.stop_condition_settings) if isinstance(self.__stop_condition_factory, MultiInstanceFactory[TStopConditionSettings, StopCondition[TrainingContext[TModel]]]) else self.__stop_condition_factory(experiment_settings.stop_condition_settings)
        objective_functions: Dict[str, ObjectiveFunction[EvaluationContext[TInput, TTarget, TModel]]] = self.__objective_function_factory.create(experiment_settings.objective_function_settings) if isinstance(self.__objective_function_factory, MultiInstanceFactory[TObjectiveFunctionSettings, ObjectiveFunction[EvaluationContext[TInput, TTarget, TModel]]]) else self.__objective_function_factory(experiment_settings.objective_function_settings)

        model = await training_service.train(model=model, datasets=training_datasets, stop_conditions=stop_conditions, objective_functions=objective_functions)

        evaluation_service: EvaluationService[TInput, TTarget, TModel, EvaluationContext[TInput, TTarget, TModel]] = self.__evaluation_service_factory.create(experiment_settings.evaluation_service_settings) if isinstance(self.__evaluation_service_factory, SingleInstanceFactory[TEvaluationServiceSettings,  EvaluationService[TInput, TTarget, TModel, EvaluationContext[TInput, TTarget, TModel]]]) else self.__evaluation_service_factory(experiment_settings.evaluation_service_settings)
        evaluation_datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]] = self.__test_dataset_factory.create(experiment_settings.evaluation_dataset_settings) if isinstance(self.__test_dataset_factory, MultiInstanceFactory[TEvaluationDatasetSettings, Dataset[Tuple[TInput, TTarget]]]) else self.__test_dataset_factory(experiment_settings.evaluation_dataset_settings)
        evaluation_metrics: Dict[str, EvaluationMetric[EvaluationContext[TInput, TTarget, TModel]]] = self.__evaluation_metric_factory.create(experiment_settings.evaluation_metric_settings) if isinstance(self.__evaluation_metric_factory, MultiInstanceFactory[TEvaluationMetricSettings, EvaluationMetric[EvaluationContext[TInput, TTarget, TModel]]]) else self.__evaluation_metric_factory(experiment_settings.evaluation_metric_settings)
        
        scores: Dict[str, Dict[str, float]] = await evaluation_service.evaluate(model=model, evaluation_datasets=evaluation_datasets, evaluation_metrics=evaluation_metrics)

        return DefaultMachineLearningExperimentResult[TModel](model, scores)

    async def __run_experiment(self, experiment_settings: Tuple[str, DefaultMachineLearningExperimentSettings]) -> Tuple[str, DefaultMachineLearningExperimentResult[TModel]]:
        result = await self.run_experiment(experiment_settings[1])

        return (experiment_settings[0], result)

    async def run_experiments(self, experiment_settings: Dict[str, DefaultMachineLearningExperimentSettings]) -> Dict[str, DefaultMachineLearningExperimentResult[TModel]]:
        experiment_tasks: List[Coroutine[Any, Any, Tuple[str, DefaultMachineLearningExperimentResult[TModel]]]] = [self.__run_experiment(settings) for settings in experiment_settings.items()]

        completed, pending = await asyncio.wait(experiment_tasks)

        return dict([t.result() for t in completed])