import asyncio
from logging import Logger
from typing import Any, Coroutine, Dict, Generic, List, Optional, Tuple
from experimentation.experiment.abstractions.experimentation_service import ExperimentationService, TExperimentSettings, TExperimentResult
from matplotlib.pyplot import eventplot
from torch.utils.data import Dataset, random_split
import nest_asyncio
from multipledispatch import dispatch

from ..evaluation.abstractions.evaluation_metric import EvaluationMetric
from ..experimentation.abstractions.stop_condition_factory import StopConditionFactory
from ..training.abstractions.stop_condition import StopCondition
from ..experimentation.abstractions.objective_function_factory import ObjectiveFunctionFactory
from ..parameter_tuning.abstractions.objective_function import ObjectiveFunction
from ..experimentation.abstractions.evaluation_metric_factory import EvaluationMetricFactory
from ..modeling.abstractions.model import TInput, TTarget
from ..evaluation.abstractions.evaluation_context import TModel, EvaluationContext
from ..training.abstractions.training_context import TrainingContext
from ..evaluation.abstractions.evaluation_service import EvaluationService, TEvaluationContext
from ..training.abstractions.training_service import TTrainingContext, TrainingService
from .default_machine_learning_experiment_result import DefaultMachineLearningExperimentResult
from .abstractions.evaluation_service_factory import EvaluationServiceFactory
from .abstractions.model_factory import ModelFactory
from .abstractions.training_service_factory import TrainingServiceFactory
from .abstractions.dataset_factory import DatasetFactoryFactory

nest_asyncio.apply()

class DefaultMachineLearningExperimentationService(Generic[TInput, TTarget, TModel], ExperimentationService[TExperimentSettings, DefaultMachineLearningExperimentResult[TModel]]):
    def __init__(self, logger: Logger,
    model_factory: ModelFactory[TModel, TExperimentSettings], 
    training_service_factory: TrainingServiceFactory[TInput, TTarget, TModel, TExperimentSettings], 
    evaluation_service_factory: EvaluationServiceFactory[TInput, TTarget, TModel, TExperimentSettings], 
    training_dataset_factory: DatasetFactoryFactory[TInput, TTarget, TExperimentSettings], 
    test_dataset_factory: DatasetFactoryFactory[TInput, TTarget, TExperimentSettings],
    evaluation_metric_factory: EvaluationMetricFactory[TInput, TTarget, TModel, TExperimentSettings],
    objective_function_factory: ObjectiveFunctionFactory[TInput, TTarget, TModel, TExperimentSettings],
    stop_condition_factory: StopConditionFactory[TModel, TExperimentSettings],
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
        self.__model_factory: ModelFactory[TModel, TExperimentSettings] = model_factory
        self.__training_service_factory: TrainingServiceFactory[TInput, TTarget, TModel, TExperimentSettings] = training_service_factory
        self.__evaluation_service_factory: EvaluationServiceFactory[TInput, TTarget, TModel, TExperimentSettings] = evaluation_service_factory
        self.__training_dataset_factory: DatasetFactoryFactory[TInput, TTarget, TExperimentSettings] = training_dataset_factory
        self.__test_dataset_factory: DatasetFactoryFactory[TInput, TTarget, TExperimentSettings] = test_dataset_factory
        self.__evaluation_metric_factory: EvaluationMetricFactory[TInput, TTarget, TModel, TExperimentSettings] = test_dataset_factory
        self.__objective_function_factory: ObjectiveFunctionFactory[TInput, TTarget, TModel, TExperimentSettings] = test_dataset_factory
        self.__stop_condition_factory: StopConditionFactory[TModel, TExperimentSettings] = test_dataset_factory

    async def run_experiment(self, experiment_settings: TExperimentSettings) -> DefaultMachineLearningExperimentResult[TModel]:
        model: TModel = self.__model_factory.create(experiment_settings)
        
        training_service: TrainingService[TInput, TTarget, TModel, TrainingContext[TModel], EvaluationContext[TInput, TTarget, TModel]] = self.__training_service_factory.create(experiment_settings)
        training_dataset: Dataset[Tuple[TInput, TTarget]] = self.__training_dataset_factory.create(experiment_settings)
        stop_conditions: Dict[str, StopCondition[TrainingContext[TModel]]] = self.__stop_condition_factory.create(experiment_settings)
        objective_functions: Dict[str, ObjectiveFunction[EvaluationContext[TInput, TTarget, TModel]]] = self.__objective_function_factory.create(experiment_settings)

        model = await training_service.train(model=model, dataset=training_dataset, stop_conditions=stop_conditions, objective_functions=objective_functions)

        evaluation_service: EvaluationService[TInput, TTarget, TModel, EvaluationContext[TInput, TTarget, TModel]] = self.__evaluation_service_factory.create(experiment_settings)
        evaluation_dataset: Dataset[Tuple[TInput, TTarget]] = self.__test_dataset_factory.create(experiment_settings)
        evaluation_metrics: Dict[str, EvaluationMetric[EvaluationContext[TInput, TTarget, TModel]]] = self.__evaluation_metric_factory.create(experiment_settings)
        
        scores: Dict[str, float] = await evaluation_service.evaluate(model=model, evaluation_dataset=evaluation_dataset, evaluation_metrics=evaluation_metrics)

        return DefaultMachineLearningExperimentResult[TModel](model, scores)

    async def __run_experiment(self, experiment_settings: Tuple[str, TExperimentSettings]) -> Tuple[str, DefaultMachineLearningExperimentResult[TModel]]:
        result = await self.run_experiment(experiment_settings[1])

        return (experiment_settings[0], result)

    async def run_experiments(self, experiment_settings: Dict[str, TExperimentSettings]) -> Dict[str, DefaultMachineLearningExperimentResult[TModel]]:
        experiment_tasks: List[Coroutine[Any, Any, Tuple[str, DefaultMachineLearningExperimentResult[TModel]]]] = [self.__run_experiment(settings) for settings in experiment_settings.items()]

        completed, pending = await asyncio.wait(experiment_tasks)

        return dict([t.result() for t in completed])