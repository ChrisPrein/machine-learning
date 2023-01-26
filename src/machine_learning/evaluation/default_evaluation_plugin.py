from abc import ABC, abstractmethod
from logging import Logger
from typing import *
from .multi_evaluation_context import MultiEvaluationContext, Score
from .evaluation_context import EvaluationContext, TModel
from ..modeling.model import TInput, TTarget, TOutput

__all__ = ['DefaultEvaluationPlugin', 'PreMultiLoop', 'PostMultiLoop', 'PreLoop', 'PostLoop', 'PreEvaluationStep', 'PostEvaluationStep', 'PreMultiEvaluationStep', 
    'PostMultiEvaluationStep']

class DefaultEvaluationPlugin(Generic[TInput, TTarget, TOutput, TModel], ABC):
    pass

class PreMultiLoop(DefaultEvaluationPlugin[TInput, TTarget, TOutput, TModel]):
    @abstractmethod
    def pre_multi_loop(self, logger: Logger, evaluation_context: MultiEvaluationContext): ...

class PostMultiLoop(DefaultEvaluationPlugin[TInput, TTarget, TOutput, TModel]):
    @abstractmethod
    def post_multi_loop(self, logger: Logger, evaluation_context: MultiEvaluationContext): ...

class PreLoop(DefaultEvaluationPlugin[TInput, TTarget, TOutput, TModel]):
    @abstractmethod
    def pre_loop(self, logger: Logger, evaluation_context: EvaluationContext[TInput, TTarget, TOutput, TModel]): ...

class PostLoop(DefaultEvaluationPlugin[TInput, TTarget, TOutput, TModel]):
    @abstractmethod
    def post_loop(self, logger: Logger, evaluation_context: EvaluationContext[TInput, TTarget, TOutput, TModel], result: Dict[str, Score]): ...

class PreEvaluationStep(DefaultEvaluationPlugin[TInput, TTarget, TOutput, TModel]):
    @abstractmethod
    def pre_evaluation_step(self, logger: Logger, evaluation_context: EvaluationContext[TInput, TTarget, TOutput, TModel]): ...

class PostEvaluationStep(DefaultEvaluationPlugin[TInput, TTarget, TOutput, TModel]):
    @abstractmethod
    def post_evaluation_step(self, logger: Logger, evaluation_context: EvaluationContext[TInput, TTarget, TOutput, TModel]): ...

class PreMultiEvaluationStep(DefaultEvaluationPlugin[TInput, TTarget, TOutput, TModel]):
    @abstractmethod
    def pre_multi_evaluation_step(self, logger: Logger, evaluation_context: MultiEvaluationContext): ...

class PostMultiEvaluationStep(DefaultEvaluationPlugin[TInput, TTarget, TOutput, TModel]):
    @abstractmethod
    def post_multi_evaluation_step(self, logger: Logger, evaluation_context: MultiEvaluationContext): ...