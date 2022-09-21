from abc import ABC, abstractmethod
from logging import Logger
from typing import *

from ...modeling.abstractions.model import *
from ..contexts.evaluation_context import *
from ..contexts.multi_evaluation_context import *

class DefaultEvaluationPlugin(Generic[TInput, TTarget, TModel], ABC):
    pass

class PreMultiLoop(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_multi_loop(self, logger: Logger, context: MultiEvaluationContext[TInput, TTarget, TModel]): ...

class PostMultiLoop(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_multi_loop(self, logger: Logger, context: MultiEvaluationContext[TInput, TTarget, TModel]): ...

class PreLoop(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_loop(self, logger: Logger, evaluationContext: EvaluationContext[TInput, TTarget, TModel]): ...

class PostLoop(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_loop(self, logger: Logger, evaluationContext: EvaluationContext[TInput, TTarget, TModel], result: Dict[str, Score]): ...

class PreEvaluationStep(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_evaluation_step(self, logger: Logger, evaluationContext: EvaluationContext[TInput, TTarget, TModel]): ...

class PostEvaluationStep(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_evaluation_step(self, logger: Logger, evaluationContext: EvaluationContext[TInput, TTarget, TModel]): ...

class PreMultiEvaluationStep(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def pre_multi_evaluation_step(self, logger: Logger, evaluationContext: MultiEvaluationContext[TInput, TTarget, TModel]): ...

class PostMultiEvaluationStep(DefaultEvaluationPlugin[TInput, TTarget, TModel]):
    @abstractmethod
    def post_multi_evaluation_step(self, logger: Logger, evaluationContext: MultiEvaluationContext[TInput, TTarget, TModel]): ...