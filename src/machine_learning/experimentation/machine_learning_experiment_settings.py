from dataclasses import dataclass
from typing import Any, Dict
from experimentation.experiment.abstractions.experiment_settings import ExperimentSettings

@dataclass    
class MachineLearningExperimentSettings(ExperimentSettings):    
    model_settings: Dict[str, Any]
    training_service_settings: Dict[str, Any]
    evaluation_service_settings: Dict[str, Any]
    evaluation_dataset_settings: Dict[str, Any]
    training_dataset_settings: Dict[str, Any]
    evaluation_metric_settings: Dict[str, Any]
    objective_function_settings: Dict[str, Any]
    stop_condition_settings: Dict[str, Any]