from ast import Call
import asyncio
import itertools
from logging import Logger
import logging
from tokenize import Single
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, Tuple, TypeVar, Union, final
from uuid import UUID
import uuid
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

START_RUN = 60
END_RUN = 61
START_EXPERIMENT = 62
END_EXPERIMENT = 63
START_EXPERIMENTS = 64
END_EXPERIMENTS = 65

EXPERIMENTATION_LOGGER_NAME = "experimentation"

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
    def __init__(self,
    model_factory: ModelFactoryAlias[TModel], 
    training_service_factory: TrainingServiceFactoryAlias[TInput, TTarget, TModel],
    evaluation_service_factory: EvaluationServiceFactoryAlias[TInput, TTarget, TModel],
    training_dataset_factory: TrainingDatasetFactoryAlias[TInput, TTarget], 
    test_dataset_factory: EvaluationDatasetFactoryAlias[TInput, TTarget],
    evaluation_metric_factory: EvaluationMetricFactoryAlias[TInput, TTarget, TModel],
    objective_function_factory: ObjectiveFunctionFactoryAlias[TInput, TTarget, TModel],
    stop_condition_factory: StopConditionFactoryAlias[TModel],
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
    logger: Optional[Logger] = None):

        if logger is None:
            self.__logger: Logger = logging.getLogger()
        else:
            self.__logger: Logger = logger.getChild(EXPERIMENTATION_LOGGER_NAME)

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

        self.__event_loop: asyncio.AbstractEventLoop = event_loop if not event_loop is None else asyncio.get_event_loop()
        self.__model_factory: ModelFactoryAlias[TModel] = model_factory
        self.__training_service_factory: TrainingServiceFactoryAlias[TInput, TTarget, TModel] = training_service_factory
        self.__evaluation_service_factory: EvaluationServiceFactoryAlias[TInput, TTarget, TModel] = evaluation_service_factory
        self.__training_dataset_factory: TrainingDatasetFactoryAlias[TInput, TTarget] = training_dataset_factory
        self.__test_dataset_factory: EvaluationDatasetFactoryAlias[TInput, TTarget] = test_dataset_factory
        self.__evaluation_metric_factory: EvaluationMetricFactoryAlias[TInput, TTarget, TModel] = evaluation_metric_factory
        self.__objective_function_factory: ObjectiveFunctionFactoryAlias[TInput, TTarget, TModel] = objective_function_factory
        self.__stop_condition_factory: StopConditionFactoryAlias[TModel] = stop_condition_factory

    async def __execute_run(self, run_settings: MachineLearningRunSettings, experiment_logger: Logger) -> MachineLearningRunResult[TModel]:
        run_id: UUID = uuid.uuid4()

        run_logger: Logger = experiment_logger.getChild(str(run_id))

        run_logger.info("executing run...")
        run_logger.log(START_RUN, run_settings)

        result: MachineLearningRunResult[TModel] = None

        try:
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

            result = MachineLearningRunResult[TModel](run_settings=run_settings, model=model, scores=scores)
        except Exception as ex:
            run_logger.error(ex)
        finally:
            run_logger.info("finished run.")
            run_logger.log(END_RUN, result)

        return result

    async def run_experiment(self, experiment_settings: MachineLearningExperimentSettings) -> MachineLearningExperimentResult[TModel]:
        experiment_logger: Logger = self.__logger.getChild(experiment_settings.name)

        combinations: List[Tuple] = itertools.product(experiment_settings.model_settings, experiment_settings.training_service_settings, 
        experiment_settings.evaluation_service_settings, experiment_settings.evaluation_dataset_settings, experiment_settings.training_dataset_settings,  
        experiment_settings.evaluation_metric_settings, experiment_settings.objective_function_settings, experiment_settings.stop_condition_settings)

        runs: List[MachineLearningRunSettings] = [MachineLearningRunSettings(*combination) for combination in combinations]

        experiment_logger.info(f"running experiment {experiment_settings.name}...")
        experiment_logger.log(START_EXPERIMENT, {"experiment_settings": experiment_settings, "runs": runs})

        result: MachineLearningExperimentResult[TModel] = None

        try:
            run_tasks: List[Coroutine[Any, Any, MachineLearningRunResult[TModel]]] = [self.__execute_run(run_settings, experiment_logger) for run_settings in runs]
            results: List[MachineLearningRunResult[TModel]] = await asyncio.gather(*run_tasks, loop=self.__event_loop)
            result = MachineLearningExperimentResult[TModel](results)
        except Exception as ex:
            experiment_logger.critical(ex)
        finally:
            experiment_logger.info(f"finished experiment {experiment_settings.name}.")
            experiment_logger.log(END_EXPERIMENT, result)

        return result

    async def __run_experiment(self, experiment_settings: Tuple[str, MachineLearningExperimentSettings]) -> Tuple[str, MachineLearningExperimentResult[TModel]]:
        result = await self.run_experiment(experiment_settings[1])

        return (experiment_settings[0], result)

    async def run_experiments(self, experiment_settings: Dict[str, MachineLearningExperimentSettings]) -> Dict[str, MachineLearningExperimentResult[TModel]]:
        self.__logger.info(f"running {len(experiment_settings)} experiments...")
        self.__logger.log(START_EXPERIMENTS, experiment_settings)
        
        experiment_tasks: List[Coroutine[Any, Any, Tuple[str, MachineLearningExperimentResult[TModel]]]] = [self.__run_experiment(settings) for settings in experiment_settings.items()]

        results: List[Tuple[str, MachineLearningExperimentResult[TModel]]] = await asyncio.gather(*experiment_tasks, loop=self.__event_loop)

        result:  Dict[str, MachineLearningExperimentResult[TModel]] = dict(results)

        self.__logger.info(f"finished all {len(experiment_settings)} experiments")
        self.__logger.log(END_EXPERIMENTS, result)

        return result