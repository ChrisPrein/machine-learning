from logging import Logger
from typing import List
from ..modeling.abstractions.model import Model, TInput, TTarget
from .abstractions.evaluation_metric import EvaluationContext, EvaluationMetric, Prediction, TModel

def default_evaluation(logger: Logger, model: TModel, input_batch: List[TInput], target_batch: List[TTarget]) -> List[TTarget]:
    return model.predict_batch(input_batch=input_batch)