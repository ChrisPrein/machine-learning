from typing import Callable
from .abstractions.evaluation_metric import EvaluationMetric, TEvaluationContext, TModel

class CustomEvaluationMetric(EvaluationMetric[TEvaluationContext]):
    def __init__(self, expression: Callable[[TEvaluationContext], float]):
        if expression is None:
            raise ValueError("expression can't be empty")

        self.expression: Callable[[TEvaluationContext], float] = expression

    def calculate_score(self, context: TEvaluationContext) -> float:
        return self.expression(context)