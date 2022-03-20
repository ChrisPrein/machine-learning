from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Tuple, Union
from torch.utils.data.dataset import Dataset

from .abstractions.model_factory import ModelFactory

from .abstractions.objective_function import ObjectiveFunction

from ..modeling.abstractions.model import TInput, TTarget
from ..evaluation.abstractions.evaluation_context import TModel
from ..evaluation.abstractions.evaluation_metric import TEvaluationContext
from .abstractions.parameter_tuning_service import ParameterTuningService

class GridSearchTuningService(ParameterTuningService[TInput, TTarget, TModel, TEvaluationContext]):
    def __init__(self):
        pass

    async def search(self, model_factory: Union[ModelFactory[TModel], Callable[[Dict[str, Any]], TModel]], params: Dict[str, List[Any]], dataset: Dataset[Tuple[TInput, TTarget]], objective_functions: Dict[str, ObjectiveFunction[TEvaluationContext]]) -> Dict[str, Any]:
        if model_factory is None:
            raise ValueError("model_factory can't be empty")

        if params is None:
            raise ValueError("params can't be empty")

        if dataset is None:
            raise ValueError("dataset can't be empty")

        if objective_functions is None:
            raise ValueError("objective_functions can't be empty")
