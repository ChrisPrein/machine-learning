from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, TypeVar
from experimentation.experiment.abstractions.experiment_settings import ExperimentSettings
from experimentation.experiment.abstractions.experiment_result import ExperimentResult
from experimentation.experiment.abstractions.experimentation_service import ExperimentationService

from ...modeling.abstractions.model import TInput, TTarget
from ...evaluation.abstractions.evaluation_metric import TModel, EvaluationContext

@dataclass
class MachineLearningExperimentResult(Generic[TModel], ExperimentResult):
    model: TModel
    scores: Dict[str, float]

@dataclass    
class MachineLearningExperimentSettings(ExperimentSettings):    
    model_settings: List[Dict[str, Any]]
    training_service_settings: List[Dict[str, Any]]
    evaluation_service_settings: List[Dict[str, Any]]
    evaluation_dataset_settings: List[Dict[str, Dict[str, Any]]]
    training_dataset_settings: List[Dict[str, Dict[str, Any]]]
    evaluation_metric_settings: List[Dict[str, Dict[str, Any]]]
    objective_function_settings: List[Dict[str, Dict[str, Any]]]
    stop_condition_settings: List[Dict[str, Dict[str, Any]]]

@dataclass    
class MachineLearningRunSettings():    
    model_settings: Dict[str, Any]
    training_service_settings: Dict[str, Any]
    evaluation_service_settings: Dict[str, Any]
    evaluation_dataset_settings: Dict[str, Dict[str, Any]]
    training_dataset_settings: Dict[str, Dict[str, Any]]
    evaluation_metric_settings: Dict[str, Dict[str, Any]]
    objective_function_settings: Dict[str, Dict[str, Any]]
    stop_condition_settings: Dict[str, Dict[str, Any]]

@dataclass 
class MachineLearningRunResult(Generic[TModel]):
    run_settings: MachineLearningRunSettings
    model: TModel
    scores: Dict[str, float]

@dataclass
class MachineLearningExperimentResult(Generic[TModel], ExperimentResult):
    run_results: List[MachineLearningRunResult[TModel]]

class MachineLearningExperimentationService(Generic[TModel], 
ExperimentationService[MachineLearningExperimentSettings, MachineLearningExperimentResult[TModel]]):
    pass