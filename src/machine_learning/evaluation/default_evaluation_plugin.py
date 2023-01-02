from abc import ABC, abstractmethod
from logging import Logger
from typing import *
from .multi_evaluation_context import MultiEvaluationContext, Score
from .evaluation_context import EvaluationContext, TModel
from ..modeling.model import TInput, TTarget

__all__ = ['DefaultEvaluationPlugin', 'PreMultiLoop', 'PostMultiLoop', 'PreLoop', 'PostLoop', 'PreEvaluationStep', 'PostEvaluationStep', 'PreMultiEvaluationStep', 
    'PostMultiEvaluationStep']

class DefaultEvaluationPlugin(Generic[TInput, TTarget, TModel], ABC):
    pass

class PreMultiLoop(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_multi_loop(self, logger: Logger, evaluation_context: MultiEvaluationContext[TInput, TTarget, TModel]): ...

class PostMultiLoop(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_multi_loop(self, logger: Logger, evaluation_context: MultiEvaluationContext[TInput, TTarget, TModel]): ...

class PreLoop(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_loop(self, logger: Logger, evaluation_context: EvaluationContext[TInput, TTarget, TModel]): ...

class PostLoop(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_loop(self, logger: Logger, evaluation_context: EvaluationContext[TInput, TTarget, TModel], result: Dict[str, Score]): ...

class PreEvaluationStep(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_evaluation_step(self, logger: Logger, evaluation_context: EvaluationContext[TInput, TTarget, TModel]): ...

class PostEvaluationStep(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_evaluation_step(self, logger: Logger, evaluation_context: EvaluationContext[TInput, TTarget, TModel]): ...

class PreMultiEvaluationStep(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_multi_evaluation_step(self, logger: Logger, evaluation_context: MultiEvaluationContext[TInput, TTarget, TModel]): ...

class PostMultiEvaluationStep(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_multi_evaluation_step(self, logger: Logger, evaluation_context: MultiEvaluationContext[TInput, TTarget, TModel]): ...