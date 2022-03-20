from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Tuple, Union
from torch.utils.data.dataset import Dataset

from .model_factory import ModelFactory

from .objective_function import ObjectiveFunction

from ...modeling.abstractions.model import TInput, TTarget
from ...evaluation.abstractions.evaluation_context import TModel
from ...evaluation.abstractions.evaluation_metric import TEvaluationContext

class ParameterTuningService(Generic[TInput, TTarget, TModel, TEvaluationContext], ABC):

    @abstractmethod
    async def search(self, model_factory: Union[ModelFactory[TModel], Callable[[Dict[str, Any]], TModel]], params: Dict[str, List[Any]], dataset: Dataset[Tuple[TInput, TTarget]], objective_functions: Dict[str, ObjectiveFunction[TEvaluationContext]]) -> Dict[str, Any]:
        pass