from dataclasses import dataclass
from typing import Any, Dict, Generic, TypeVar
from experimentation.experiment.abstractions.experiment_settings import ExperimentSettings

TModelSettings = TypeVar("TModelSettings")
TTrainingServiceSettings = TypeVar("TTrainingServiceSettings")
TEvaluationServiceSettings = TypeVar("TEvaluationServiceSettings")
TEvaluationDatasetSettings = TypeVar("TEvaluationDatasetSettings")
TTrainingDatasetSettings = TypeVar("TTrainingDatasetSettings")
TEvaluationMetricSettings = TypeVar("TEvaluationMetricSettings")
TObjectiveFunctionSettings = TypeVar("TObjectiveFunctionSettings")
TStopConditionSettings = TypeVar("TStopConditionSettings")

@dataclass    
class DefaultMachineLearningExperimentSettings(Generic[TModelSettings, TTrainingServiceSettings, TEvaluationServiceSettings, TEvaluationDatasetSettings, TTrainingDatasetSettings, 
TEvaluationMetricSettings, TObjectiveFunctionSettings, TStopConditionSettings], ExperimentSettings):    
    model_settings: TModelSettings
    training_service_settings: TTrainingServiceSettings
    evaluation_service_settings: TEvaluationServiceSettings
    evaluation_dataset_settings: TEvaluationDatasetSettings
    training_dataset_settings: TTrainingDatasetSettings
    evaluation_metric_settings: TEvaluationMetricSettings
    objective_function_settings: TObjectiveFunctionSettings
    stop_condition_settings: TStopConditionSettings