from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, Union
from sklearn.model_selection import GridSearchCV
from torch.utils.data.dataset import Dataset
from sklearn import metrics

from ..modeling.adapter.sklearn_estimator_adapter import SkleanEstimatorAdapter
from .abstractions.model_factory import ModelFactory
from .abstractions.objective_function import ObjectiveFunction, OptimizationType
from ..modeling.abstractions.model import TInput, TTarget
from ..evaluation.abstractions.evaluation_context import TModel
from ..evaluation.abstractions.evaluation_metric import TEvaluationContext
from .abstractions.parameter_tuning_service import ParameterTuningService

class GridSearchTuningService(ParameterTuningService[TInput, TTarget, TModel, TEvaluationContext]):
    def __init__(self, folds: Optional[int] = 5):
        self.__folds: int = folds

    async def search(self, model_factory: Union[ModelFactory[TModel], Callable[[Dict[str, Any]], TModel]], params: Dict[str, List[Any]], dataset: Dataset[Tuple[TInput, TTarget]], objective_functions: Dict[str, ObjectiveFunction[TEvaluationContext]]) -> Dict[str, Any]:
        if model_factory is None:
            raise ValueError("model_factory can't be empty")

        if params is None:
            raise ValueError("params can't be empty")

        if dataset is None:
            raise ValueError("dataset can't be empty")

        if objective_functions is None:
            raise ValueError("objective_functions can't be empty")

        factory_params: Dict[str, Any] = {key: values[0] for key, values in params.items()}
        model: TModel = model_factory.create(factory_params)
        estimator = SkleanEstimatorAdapter(model)
        scorer = {key: metrics.make_scorer(objective_function.calculate_score, greater_is_better=objective_function.optimization_type == OptimizationType.MAX) for key, objective_function in objective_functions.items()}

        inputs: List[TInput] = [input for (input, target) in dataset]
        targets: List[TTarget] = [target for (input, target) in dataset]

        grid_search = GridSearchCV(estimator=estimator, param_grid=params, cv=self.folds, scoring=scorer)
        grid_search.fit(X=inputs, y=targets)

        return grid_search.best_params_

