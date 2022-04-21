from ast import Call
import asyncio
from logging import Logger
from tokenize import Single
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, Tuple, TypeVar, Union
from experimentation.experiment.abstractions.experimentation_service import ExperimentationService, TExperimentSettings, TExperimentResult
from matplotlib.pyplot import eventplot
from torch.utils.data import Dataset, random_split
import nest_asyncio
from multipledispatch import dispatch

from .abstractions.instance_factory import InstanceFactory
from ..evaluation.abstractions.evaluation_metric import EvaluationMetric
from ..training.abstractions.stop_condition import StopCondition
from ..parameter_tuning.abstractions.objective_function import ObjectiveFunction
from ..modeling.abstractions.model import TInput, TTarget
from ..evaluation.abstractions.evaluation_context import TModel, EvaluationContext
from ..training.abstractions.training_context import TrainingContext
from ..evaluation.abstractions.evaluation_service import EvaluationService, TEvaluationContext
from ..training.abstractions.training_service import TTrainingContext, TrainingService
from .abstractions.machine_learning_experimentation_service import MachineLearningExperimentationService, MachineLearningExperimentSettings, MachineLearningExperimentResult, MachineLearningRunResult, MachineLearningRunSettings

nest_asyncio.apply()

TSettings = TypeVar('TSettings')
TInstance = TypeVar('TInstance')

TrainingServiceAlias = TrainingService[TInput, TTarget, TModel, TrainingContext[TModel], EvaluationContext[TInput, TTarget, TModel]]
EvaluationServiceAlias = EvaluationService[TInput, TTarget, TModel, EvaluationContext[TInput, TTarget, TModel]]
EvaluationMetricAlias = EvaluationMetric[EvaluationContext[TInput, TTarget, TModel]]
ObjectiveFunctionAlias = ObjectiveFunction[EvaluationContext[TInput, TTarget, TModel]]
StopConditionAlias = StopCondition[TrainingContext[TModel]]

FactoryAlias = Union[InstanceFactory[TSettings, TInstance], Callable[[TSettings], TInstance]]
ModelFactoryAlias = FactoryAlias[TModelSettings, TModel]
TrainingServiceFactoryAlias = FactoryAlias[TTrainingServiceSettings, TrainingServiceAlias[TInput, TTarget, TModel]]
EvaluationServiceFactoryAlias = FactoryAlias[TEvaluationServiceSettings, EvaluationServiceAlias[TInput, TTarget, TModel]]
TrainingDatasetFactoryAlias = FactoryAlias[TTrainingDatasetSettings, Dict[str, Dataset[Tuple[TInput, TTarget]]]]
EvaluationDatasetFactoryAlias = FactoryAlias[TEvaluationDatasetSettings, Dict[str, Dataset[Tuple[TInput, TTarget]]]]
EvaluationMetricFactoryAlias = FactoryAlias[TEvaluationMetricSettings, Dict[str, EvaluationMetricAlias[TInput, TTarget, TModel]]]
ObjectiveFunctionFactoryAlias = FactoryAlias[TObjectiveFunctionSettings, Dict[str, ObjectiveFunctionAlias[TInput, TTarget, TModel]]]
StopConditionFactoryAlias = FactoryAlias[TStopConditionSettings, Dict[str, StopConditionAlias[TModel]]]

class DefaultMachineLearningExperimentationService(MachineLearningExperimentationService[TModel]):
    def __init__(self, logger: Logger,
    model_factory: ModelFactoryAlias[TModelSettings, TModel], 
    training_service_factory: TrainingServiceFactoryAlias[TTrainingServiceSettings, TInput, TTarget, TModel],
    evaluation_service_factory: EvaluationServiceFactoryAlias[TEvaluationServiceSettings, TInput, TTarget, TModel],
    training_dataset_factory: TrainingDatasetFactoryAlias[TTrainingDatasetSettings, TInput, TTarget], 
    test_dataset_factory: EvaluationDatasetFactoryAlias[TEvaluationDatasetSettings, TInput, TTarget],
    evaluation_metric_factory: EvaluationMetricFactoryAlias[TEvaluationMetricSettings, TInput, TTarget, TModel],
    objective_function_factory: ObjectiveFunctionFactoryAlias[TObjectiveFunctionSettings, TInput, TTarget, TModel],
    stop_condition_factory: StopConditionFactoryAlias[TStopConditionSettings, TModel],
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
        self.__model_factory: ModelFactoryAlias[TModelSettings, TModel] = model_factory
        self.__training_service_factory: TrainingServiceFactoryAlias[TTrainingServiceSettings, TInput, TTarget, TModel] = training_service_factory
        self.__evaluation_service_factory: EvaluationServiceFactoryAlias[TEvaluationServiceSettings, TInput, TTarget, TModel] = evaluation_service_factory
        self.__training_dataset_factory: TrainingDatasetFactoryAlias[TTrainingDatasetSettings, TInput, TTarget] = training_dataset_factory
        self.__test_dataset_factory: EvaluationDatasetFactoryAlias[TEvaluationDatasetSettings, TInput, TTarget] = test_dataset_factory
        self.__evaluation_metric_factory: EvaluationMetricFactoryAlias[TEvaluationMetricSettings, TInput, TTarget, TModel] = evaluation_metric_factory
        self.__objective_function_factory: ObjectiveFunctionFactoryAlias[TObjectiveFunctionSettings, TInput, TTarget, TModel] = objective_function_factory
        self.__stop_condition_factory: StopConditionFactoryAlias[TStopConditionSettings, TModel] = stop_condition_factory

    async def __execute_run(self, run_settings: MachineLearningRunSettings[TModelSettings, TTrainingServiceSettings, TEvaluationServiceSettings, TEvaluationDatasetSettings, TTrainingDatasetSettings, 
TEvaluationMetricSettings, TObjectiveFunctionSettings, TStopConditionSettings]) -> MachineLearningRunResult[TModel, TModelSettings, TTrainingServiceSettings, TEvaluationServiceSettings, TTrainingDatasetSettings, TEvaluationDatasetSettings, TEvaluationMetricSettings, TObjectiveFunctionSettings, TStopConditionSettings]:
        model: TModel = self.__model_factory(run_settings.model_settings)

        training_service: TrainingServiceAlias[TInput, TTarget, TModel] = self.__training_service_factory(run_settings.training_service_settings)
        training_datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]] = self.__training_dataset_factory(run_settings.training_dataset_settings)
        stop_conditions: Dict[str, StopConditionAlias[TModel]] =  self.__stop_condition_factory(run_settings.stop_condition_settings)
        objective_functions: Dict[str, ObjectiveFunctionAlias[TInput, TTarget, TModel]] = self.__objective_function_factory(run_settings.objective_function_settings)

        model = await training_service.train(model=model, datasets=training_datasets, stop_conditions=stop_conditions, objective_functions=objective_functions)

        evaluation_service: EvaluationServiceAlias[TInput, TTarget, TModel] = self.__evaluation_service_factory(run_settings.evaluation_service_settings)
        evaluation_datasets: Dict[str, Dataset[Tuple[TInput, TTarget]]] = self.__test_dataset_factory(run_settings.evaluation_dataset_settings)
        evaluation_metrics: Dict[str, EvaluationMetricAlias[TInput, TTarget, TModel]] = self.__evaluation_metric_factory(run_settings.evaluation_metric_settings)
        
        scores: Dict[str, Dict[str, float]] = await evaluation_service.evaluate(model=model, evaluation_datasets=evaluation_datasets, evaluation_metrics=evaluation_metrics)

        return MachineLearningRunResult[TModel, TModelSettings, TTrainingServiceSettings, TEvaluationServiceSettings, TTrainingDatasetSettings, TEvaluationDatasetSettings, TEvaluationMetricSettings, TObjectiveFunctionSettings, TStopConditionSettings](run_settings=run_settings, model=model, scores=scores)

    async def run_experiment(self, experiment_settings: MachineLearningExperimentSettings[TModelSettings, TTrainingServiceSettings, TEvaluationServiceSettings, TEvaluationDatasetSettings, TTrainingDatasetSettings, 
TEvaluationMetricSettings, TObjectiveFunctionSettings, TStopConditionSettings]) -> MachineLearningExperimentResult[TModel, TModelSettings, TTrainingServiceSettings, TEvaluationServiceSettings, TEvaluationDatasetSettings, TTrainingDatasetSettings, 
TEvaluationMetricSettings, TObjectiveFunctionSettings, TStopConditionSettings]:
        zipped_settings: List[Tuple[TModelSettings, TTrainingServiceSettings, TEvaluationServiceSettings, TTrainingDatasetSettings, TEvaluationDatasetSettings, TEvaluationMetricSettings, TObjectiveFunctionSettings, TStopConditionSettings]] = zip(experiment_settings.model_settings, experiment_settings.training_service_settings, experiment_settings.evaluation_service_settings, experiment_settings.training_dataset_settings, experiment_settings.evaluation_dataset_settings, experiment_settings.evaluation_metric_settings, experiment_settings.objective_function_settings, experiment_settings.stop_condition_settings)

        run_settings: List[MachineLearningRunSettings] = [MachineLearningRunSettings(zipped_item[0], zipped_item[1], zipped_item[2, zipp]) for zipped_item in zipped_settings]

        run_tasks: List[Coroutine[Any, Any, MachineLearningRunResult[TModel, TModelSettings, TTrainingServiceSettings, TEvaluationServiceSettings, TEvaluationDatasetSettings, TTrainingDatasetSettings, 
TEvaluationMetricSettings, TObjectiveFunctionSettings, TStopConditionSettings]]] = [self.__execute_run()]

        return MachineLearningExperimentResult[TModel, TModelSettings, TTrainingServiceSettings, TEvaluationServiceSettings, TEvaluationDatasetSettings, TTrainingDatasetSettings, 
TEvaluationMetricSettings, TObjectiveFunctionSettings, TStopConditionSettings](model, scores)

    async def __run_experiment(self, experiment_settings: Tuple[str, MachineLearningExperimentSettings[TModelSettings, TTrainingServiceSettings, TEvaluationServiceSettings, TEvaluationDatasetSettings, TTrainingDatasetSettings, 
TEvaluationMetricSettings, TObjectiveFunctionSettings, TStopConditionSettings]]) -> Tuple[str, MachineLearningExperimentResult[TModel, TModelSettings, TTrainingServiceSettings, TEvaluationServiceSettings, TEvaluationDatasetSettings, TTrainingDatasetSettings, 
TEvaluationMetricSettings, TObjectiveFunctionSettings, TStopConditionSettings]]:
        result = await self.run_experiment(experiment_settings[1])

        return (experiment_settings[0], result)

    async def run_experiments(self, experiment_settings: Dict[str, MachineLearningExperimentSettings[TModelSettings, TTrainingServiceSettings, TEvaluationServiceSettings, TEvaluationDatasetSettings, TTrainingDatasetSettings, 
TEvaluationMetricSettings, TObjectiveFunctionSettings, TStopConditionSettings]]) -> Dict[str, MachineLearningExperimentResult[TModel, TModelSettings, TTrainingServiceSettings, TEvaluationServiceSettings, TEvaluationDatasetSettings, TTrainingDatasetSettings, 
TEvaluationMetricSettings, TObjectiveFunctionSettings, TStopConditionSettings]]:
        experiment_tasks: List[Coroutine[Any, Any, Tuple[str, MachineLearningExperimentResult[TModel, TModelSettings, TTrainingServiceSettings, TEvaluationServiceSettings, TEvaluationDatasetSettings, TTrainingDatasetSettings, 
TEvaluationMetricSettings, TObjectiveFunctionSettings, TStopConditionSettings]]]] = [self.__run_experiment(settings) for settings in experiment_settings.items()]

        completed, pending = await asyncio.wait(experiment_tasks)

        return dict([t.result() for t in completed])