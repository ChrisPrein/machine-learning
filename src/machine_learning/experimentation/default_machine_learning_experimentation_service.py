from typing import Any, Coroutine, Dict, Generic, Tuple
from experimentation.experiment.abstractions.experimentation_service import ExperimentationService, TExperimentSettings, TExperimentResult
from torch.utils.data import Dataset, random_split

from ..experimentation.abstractions.stop_condition_factory import StopConditionFactory
from ..training.abstractions.stop_condition import StopCondition

from ..experimentation.abstractions.objective_function_factory import ObjectiveFunctionFactory

from ..parameter_tuning.abstractions.objective_function import ObjectiveFunction

from ..experimentation.abstractions.evaluation_metric_factory import EvaluationMetricFactory

from ..modeling.abstractions.model import TInput, TTarget
from ..evaluation.abstractions.evaluation_context import TModel, EvaluationContext
from ..training.abstractions.training_context import TrainingContext
from ..evaluation.abstractions.evaluation_service import TEvaluationContext
from ..training.abstractions.training_service import TTrainingContext, TrainingService
from .default_machine_learning_experiment_result import DefaultMachineLearningExperimentResult
from .abstractions.evaluation_service_factory import EvaluationServiceFactory
from .abstractions.model_factory import ModelFactory
from .abstractions.training_service_factory import TrainingServiceFactory
from .abstractions.dataset_factory import DatasetFactoryFactory

class DefaultMachineLearningExperimentationService(Generic[TInput, TTarget, TModel], ExperimentationService[TExperimentSettings, DefaultMachineLearningExperimentResult[TModel]]):
    def __init__(self, model_factory: ModelFactory[TModel, TExperimentSettings], 
    training_service_factory: TrainingServiceFactory[TInput, TTarget, TModel, TExperimentSettings], 
    evaluation_service_factory: EvaluationServiceFactory[TInput, TTarget, TModel, TExperimentSettings], 
    training_dataset_factory: DatasetFactoryFactory[TInput, TTarget, TExperimentSettings], 
    test_dataset_factory: DatasetFactoryFactory[TInput, TTarget, TExperimentSettings],
    evaluation_metric_factory: EvaluationMetricFactory[TInput, TTarget, TModel, TExperimentSettings],
    objective_function_factory: ObjectiveFunctionFactory[TInput, TTarget, TModel, TExperimentSettings],
    stop_condition_factory: StopConditionFactory[TModel, TExperimentSettings]):

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

        training_coroutine: Coroutine[Any, Any, TModel] = training_service.train(model=model, dataset=training_dataset, stop_conditions=stop_conditions, objective_functions=objective_functions)

    async def run_experiments(self, experiment_settings: Dict[str, TExperimentSettings]) -> Dict[str, DefaultMachineLearningExperimentResult[TModel]]:
        pass