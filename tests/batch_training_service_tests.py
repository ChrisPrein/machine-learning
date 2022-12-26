import asyncio
import unittest
from unittest.mock import MagicMock, Mock, patch
from typing import Any, Coroutine, List, Dict, Tuple
from faker import Faker
import random

from src.machine_learning.modeling.model import *
from src.machine_learning.training.batch_training_service import *
from src.machine_learning.evaluation.evaluation_metric import *

class BatchTrainingServiceTestCase(unittest.TestCase):
    def setUp(self):
        fake = Faker()

        self.samples: List[Tuple[str, str]] = [(fake.first_name(), fake.last_name()) for i in range(100)]

        self.model: Model[str, str] = MagicMock(spec=Model)
        self.model.predict_step = Mock(return_value=[fake.last_name() for i in range(10)])
        self.model.training_step = Mock(return_value=fake.pyfloat(positive=True))

        self.objective_function_1: EvaluationMetric[str, str] = MagicMock(spec=EvaluationMetric)
        self.objective_function_1.score = Mock(return_value=fake.pyfloat(positive=True))

        self.objective_function_2: EvaluationMetric[str, str] = MagicMock(spec=EvaluationMetric)
        self.objective_function_2.score = Mock(return_value=fake.pyfloat(positive=True))

        self.data: List[List[Tuple[str, str]]] = [random.choices(self.samples, k=2) for i in range(0, 100)]

        self.event_loop = asyncio.get_event_loop()

    def tearDown(self):
        pass

    def test_train_valid_objectives_and_dataset_should_return_trained_model(self):
        training_service: BatchTrainingService[str, str, Model[str, str]] = BatchTrainingService[str, str, Model[str, str]]()

        training_routine: Coroutine[Any, Any, Model[str, str]] = training_service.train(self.model, ("test", self.data), None)

        trained_model: Model[str, str] = self.event_loop.run_until_complete(training_routine)

    def test_train_valid_objectives_and_datasets_should_call_plugin_methods(self):
        pre_loop: PreLoop[str, str, Model[str, str]] = MagicMock(spec=PreLoop)
        pre_loop.pre_loop = Mock()
        post_loop: PostLoop[str, str, Model[str, str]] = MagicMock(spec=PostLoop)
        post_loop.post_loop = Mock()
        pre_epoch: PreEpoch[str, str, Model[str, str]] = MagicMock(spec=PreEpoch)
        pre_epoch.pre_epoch = Mock()
        post_epoch: PostEpoch[str, str, Model[str, str]] = MagicMock(spec=PostEpoch)
        post_epoch.post_epoch = Mock()
        pre_train_step: PreTrain[str, str, Model[str, str]] = MagicMock(spec=PreTrain)
        pre_train_step.pre_train = Mock()
        post_train_step: PostTrain[str, str, Model[str, str]] = MagicMock(spec=PostTrain)
        post_train_step.post_train = Mock()

        plugins: Dict[str, BatchTrainingPlugin[TInput, TTarget, TModel]] = {'pre_loop': pre_loop, 'post_loop': post_loop,
        'pre_epoch': pre_epoch, 'post_epoch': post_epoch, 'pre_train_step': pre_train_step, 'post_train_step': post_train_step}

        training_service: BatchTrainingService[str, str, Model[str, str]] = BatchTrainingService[str, str, Model[str, str]](plugins=plugins)

        datasets: Tuple[str, Iterable[Iterable[Tuple[str, str]]]] = ("set_1", self.data)

        training_routine: Coroutine[Any, Any, Model[str, str]] = training_service.train(self.model, datasets, None)

        trained_model: Model[str, str] = self.event_loop.run_until_complete(training_routine)

        pre_loop.pre_loop.assert_called()
        post_loop.post_loop.assert_called()
        pre_epoch.pre_epoch.assert_called()
        post_epoch.post_epoch.assert_called()
        pre_train_step.pre_train.assert_called()
        post_train_step.post_train.assert_called()