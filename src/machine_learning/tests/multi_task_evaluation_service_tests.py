from asyncio import coroutine
import asyncio
from dataclasses import dataclass
from re import S
import unittest
from unittest.mock import Mock, patch
from torch.utils.data import Dataset
from dataset_handling.dataloader import DataLoader
from typing import Any, Coroutine, List, Dict, Tuple
from faker import Faker
import random
from ..evaluation.abstractions.evaluation_metric import EvaluationMetric
from ..evaluation.multi_task_evaluation_service import MultiTaskEvaluationService
from ..modeling.abstractions.model import Model, TInput, TTarget

class MultiTaskEvaluationServiceTestCase(unittest.TestCase):
    def setUp(self):
        config = {'__getitem__'}
        fake = Faker()

        self.samples: List[Tuple[str, str]] = [(fake.first_name(), fake.last_name()) for i in range(10)]

        self.model_patcher = patch('machine_learning.modeling.abstractions.model.Model', new=Model[str, str])
        self.evaluation_metric_1_patcher = patch('machine_learning.evaluation.abstractions.evaluation_metric.EvaluationMetric', new=EvaluationMetric[str])
        self.evaluation_metric_2_patcher = patch('machine_learning.evaluation.abstractions.evaluation_metric.EvaluationMetric', new=EvaluationMetric[str])
        self.datalaoder_patcher = patch('dataset_handling.dataloader.DataLoader', new=DataLoader[Tuple[str, str]])

        self.model: Model[str, str] = self.model_patcher.start()

        self.model.predict_batch = Mock(return_value=[fake.last_name() for i in range(10)])

        self.evaluation_metric_1: EvaluationMetric[str] = self.evaluation_metric_1_patcher.start()

        self.evaluation_metric_1.calculate_score = Mock(return_value=fake.pyfloat(positive=True))

        self.evaluation_metric_2: EvaluationMetric[str] = self.evaluation_metric_2_patcher.start()

        self.evaluation_metric_2.calculate_score = Mock(return_value=fake.pyfloat(positive=True))

        self.dataset_patcher = patch('torch.utils.data.Dataset')
        self.dataset: Dataset[Tuple[str, str]] = self.dataset_patcher.start()
        self.dataset.__getitem__ = Mock(return_value=random.choice(self.samples))
        self.dataset.__len__ = Mock(return_value=self.samples.__len__())

        self.dataloader: DataLoader[Tuple[str, str]] = DataLoader[Tuple[str, str]](self.dataset, batch_size=1, shuffle=True)

        self.event_loop = asyncio.get_event_loop()

        self.evaluation_service: MultiTaskEvaluationService[str, str, Model[str, str]] = MultiTaskEvaluationService[str, str, Model[str, str]](event_loop=self.event_loop)

    def tearDown(self):
        self.datalaoder_patcher.stop()

    def test_evaluate_valid_model_metrics_and_dataloader_should_return_results_for_each_metric(self):
        evaluation_routine: Coroutine[Any, Any, Dict[str, float]] = self.evaluation_service.evaluate(model = self.model, evaluation_data_loader=self.dataloader, 
                    evaluation_metrics={'metric 1': self.evaluation_metric_1, 'metric 2': self.evaluation_metric_2})

        result: Dict[str, float] = self.event_loop.run_until_complete(evaluation_routine)

        assert len(result.items()) == 2