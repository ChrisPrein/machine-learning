from ast import Call
from typing import Callable, Optional, Union
from .abstractions.objective_function import ObjectiveFunction, OptimizationType
from ..evaluation.abstractions.evaluation_metric import EvaluationMetric, TEvaluationContext


class CustomObjectiveFunction(ObjectiveFunction[TEvaluationContext]):
    def __init__(self, expression: Union[Callable[[TEvaluationContext], float], EvaluationMetric[TEvaluationContext]], optimization_type: Optional[OptimizationType] = OptimizationType.MAX):
        if expression is None:
            raise ValueError("expression can't be empty")

        self.expression: Callable[[TEvaluationContext], float]

        if isinstance(expression, Callable[[TEvaluationContext], float]):
            self.expression = expression
        else:
            self.expression = expression.calculate_score

        self.__optimization_type: OptimizationType = optimization_type

    @property
    def optimization_type(self) -> OptimizationType:
        return self.__optimization_type