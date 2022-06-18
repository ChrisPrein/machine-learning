from asyncio import coroutine
import asyncio
from dataclasses import dataclass
from re import S
import unittest
from unittest.mock import MagicMock, Mock, patch
from torch import isin
from torch.utils.data import Dataset
from dataset_handling.dataloader import DataLoader
from typing import Any, Coroutine, List, Dict, Tuple
from faker import Faker
import random
from ..evaluation.abstractions.evaluation_metric import EvaluationMetric
from ..modeling.abstractions.model import Model, TInput, TTarget
from ..evaluation.default_evaluation_service import *
from ..evaluation.abstractions.default_evaluation_plugin import *

class DefaultEvaluationServiceTestCase(unittest.TestCase):
    def setUp(self):
        fake = Faker()

        self.samples: List[Tuple[str, str]] = [(fake.first_name(), fake.last_name()) for i in range(10)]

        self.model: Model[str, str] = MagicMock(spec=Model)

        self.model.predict_step = Mock(return_value=[fake.last_name() for i in range(10)])

        self.evaluation_metric_1: EvaluationMetric[str, str] = MagicMock(spec=EvaluationMetric)
        self.evaluation_metric_1.score = Mock(return_value=fake.pyfloat(positive=True))

        self.evaluation_metric_2: EvaluationMetric[str, str] = MagicMock(spec=EvaluationMetric)
        self.evaluation_metric_2.score = Mock(return_value=fake.pyfloat(positive=True))

        self.dataset: Dataset[Tuple[str, str]] = Mock()
        self.dataset.__getitem__ = Mock(return_value=random.choice(self.samples))
        self.dataset.__len__ = Mock(return_value=self.samples.__len__())

        self.event_loop = asyncio.get_event_loop()

    def tearDown(self):
        pass

    def test_evaluate_valid_model_metrics_and_dataset_should_return_results_for_each_metric(self):
        evaluation_service: DefaultEvaluationService[str, str, Model[str, str]] = DefaultEvaluationService[str, str, Model[str, str]]()

        evaluation_routine: Coroutine[Any, Any, Dict[str, float]] = evaluation_service.evaluate(self.model, self.dataset, 
                    {'metric 1': self.evaluation_metric_1, 'metric 2': self.evaluation_metric_2})

        result: Dict[str, float] = self.event_loop.run_until_complete(evaluation_routine)

        assert len(result.items()) == 2

    def test_evaluation_on_multiple_datasets_valid_model_metrics_and_datasets_should_return_results_for_each_metric_on_each_dataset(self):
        evaluation_service: DefaultEvaluationService[str, str, Model[str, str]] = DefaultEvaluationService[str, str, Model[str, str]]()

        datasets: Dict[str, Dataset[Tuple[str, str]]] = {"set_1": self.dataset, "set_2": self.dataset}

        evaluation_routine: Coroutine[Any, Any, Dict[str, Dict[str, float]]] = evaluation_service.evaluate_on_multiple_datasets(self.model, datasets, 
            {'metric 1': self.evaluation_metric_1, 'metric 2': self.evaluation_metric_2})

        result: Dict[str, Dict[str, float]] = self.event_loop.run_until_complete(evaluation_routine)

        assert len(result.items()) == 2

    def test_evaluation_on_multiple_datasets_valid_model_metrics_and_datasets_should_call_plugin_methods(self):
        pre_multi_loop: PreMultiLoop[str, str, Model[str, str]] = MagicMock(spec=PreMultiLoop)
        pre_multi_loop.pre_multi_loop = Mock()
        post_multi_loop: PostMultiLoop[str, str, Model[str, str]] = MagicMock(spec=PostMultiLoop)
        post_multi_loop.post_multi_loop = Mock()
        pre_multi_evaluation_step: PreMultiEvaluationStep[str, str, Model[str, str]] = MagicMock(spec=PreMultiEvaluationStep)
        pre_multi_evaluation_step.pre_multi_evaluation_step = Mock()
        post_multi_evaluation_step: PostMultiEvaluationStep[str, str, Model[str, str]] = MagicMock(spec=PostMultiEvaluationStep)
        post_multi_evaluation_step.post_multi_evaluation_step = Mock()
        pre_loop: PreLoop[str, str, Model[str, str]] = MagicMock(spec=PreLoop)
        pre_loop.pre_loop = Mock()
        post_loop: PostLoop[str, str, Model[str, str]] = MagicMock(spec=PostLoop)
        post_loop.post_loop = Mock()
        pre_evaluation_step: PreEvaluationStep[str, str, Model[str, str]] = MagicMock(spec=PreEvaluationStep)
        pre_evaluation_step.pre_evaluation_step = Mock()
        post_evaluation_step: PostEvaluationStep[str, str, Model[str, str]] = MagicMock(spec=PostEvaluationStep)
        post_evaluation_step.post_evaluation_step = Mock()

        plugins: Dict[str, DefaultEvaluationPlugin[TInput, TTarget, TModel]] = {'pre_multi_loop': pre_multi_loop, 'post_multi_loop': post_multi_loop, 
        'pre_multi_train_step': pre_multi_evaluation_step, 'post_multi_train_step': post_multi_evaluation_step, 'pre_loop': pre_loop, 'post_loop': post_loop,
        'pre_train_step': pre_evaluation_step, 'post_train_step': post_evaluation_step}

        evaluation_service: DefaultEvaluationService[str, str, Model[str, str]] = DefaultEvaluationService[str, str, Model[str, str]](plugins=plugins)

        datasets: Dict[str, Dataset[Tuple[str, str]]] = {"set_1": self.dataset, "set_2": self.dataset}

        evaluation_routine: Coroutine[Any, Any, Dict[str, Dict[str, float]]] = evaluation_service.evaluate_on_multiple_datasets(self.model, datasets, 
            {'metric 1': self.evaluation_metric_1, 'metric 2': self.evaluation_metric_2})

        result: Dict[str, Dict[str, float]] = self.event_loop.run_until_complete(evaluation_routine)

        pre_multi_loop.pre_multi_loop.assert_called()
        post_multi_loop.post_multi_loop.assert_called()
        pre_multi_evaluation_step.pre_multi_evaluation_step.assert_called()
        post_multi_evaluation_step.post_multi_evaluation_step.assert_called()
        pre_loop.pre_loop.assert_called()
        post_loop.post_loop.assert_called()
        pre_evaluation_step.pre_evaluation_step.assert_called()
        post_evaluation_step.post_evaluation_step.assert_called()