import asyncio
import unittest
from unittest.mock import MagicMock, Mock, patch
from torch.utils.data import Dataset
from typing import Any, Coroutine, List, Dict, Tuple
from faker import Faker
import random

from ..modeling.abstractions.model import *
from ..training.batch_training_service import *
from ..training.abstractions.objective_function import *
from ..training.abstractions.batch_training_plugin import *

class BatchTrainingServiceTestCase(unittest.TestCase):
    def setUp(self):
        fake = Faker()

        self.samples: List[Tuple[str, str]] = [(fake.first_name(), fake.last_name()) for i in range(100)]

        self.model: Model[str, str] = MagicMock(spec=Model)
        self.model.predict_step = Mock(return_value=[fake.last_name() for i in range(10)])
        self.model.training_step = Mock(return_value=fake.pyfloat(positive=True))

        self.objective_function_1: ObjectiveFunction[str, str] = MagicMock(spec=ObjectiveFunction)
        self.objective_function_1.score = Mock(return_value=fake.pyfloat(positive=True))

        self.objective_function_2: ObjectiveFunction[str, str] = MagicMock(spec=ObjectiveFunction)
        self.objective_function_2.score = Mock(return_value=fake.pyfloat(positive=True))

        self.dataset: Dataset[Tuple[str, str]] = Mock()
        self.dataset.__getitem__ = Mock(return_value=random.choice(self.samples))
        self.dataset.__len__ = Mock(return_value=self.samples.__len__())

        self.event_loop = asyncio.get_event_loop()

    def tearDown(self):
        pass

    def test_train_valid_objectives_and_dataset_should_return_trained_model(self):
        training_service: BatchTrainingService[str, str, Model[str, str]] = BatchTrainingService[str, str, Model[str, str]]()

        training_routine: Coroutine[Any, Any, Model[str, str]] = training_service.train(self.model, ("test", self.dataset), {}, 
                    {'objective 1': self.objective_function_1, 'objective 2': self.objective_function_2}, None)

        trained_model: Model[str, str] = self.event_loop.run_until_complete(training_routine)

    def test_train_on_multiple_datasets_valid_objectives_and_datasets_should_return_trained_model(self):
        training_service: BatchTrainingService[str, str, Model[str, str]] = BatchTrainingService[str, str, Model[str, str]]()

        datasets: Dict[str, Dataset[Tuple[str, str]]] = {"set_1": self.dataset, "set_2": self.dataset}

        training_routine: Coroutine[Any, Any, Model[str, str]] = training_service.train_on_multiple_datasets(self.model, datasets, {},
            {'objective 1': self.objective_function_1, 'objective 2': self.objective_function_2}, None)

        trained_model: Model[str, str] = self.event_loop.run_until_complete(training_routine)

    def test_train_on_multiple_datasets_valid_objectives_and_datasets_should_call_plugin_methods(self):
        pre_multi_loop: PreMultiLoop[str, str, Model[str, str]] = MagicMock(spec=PreMultiLoop)
        pre_multi_loop.pre_multi_loop = Mock()
        post_multi_loop: PostMultiLoop[str, str, Model[str, str]] = MagicMock(spec=PostMultiLoop)
        post_multi_loop.post_multi_loop = Mock()
        pre_multi_train_step: PreMultiTrainStep[str, str, Model[str, str]] = MagicMock(spec=PreMultiTrainStep)
        pre_multi_train_step.pre_multi_train_step = Mock()
        post_multi_train_step: PostMultiTrainStep[str, str, Model[str, str]] = MagicMock(spec=PostMultiTrainStep)
        post_multi_train_step.post_multi_train_step = Mock()
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

        plugins: Dict[str, BatchTrainingPlugin[TInput, TTarget, TModel]] = {'pre_multi_loop': pre_multi_loop, 'post_multi_loop': post_multi_loop, 
        'pre_multi_train_step': pre_multi_train_step, 'post_multi_train_step': post_multi_train_step, 'pre_loop': pre_loop, 'post_loop': post_loop,
        'pre_epoch': pre_epoch, 'post_epoch': post_epoch, 'pre_train_step': pre_train_step, 'post_train_step': post_train_step}

        training_service: BatchTrainingService[str, str, Model[str, str]] = BatchTrainingService[str, str, Model[str, str]](plugins=plugins)

        datasets: Dict[str, Dataset[Tuple[str, str]]] = {"set_1": self.dataset, "set_2": self.dataset}

        training_routine: Coroutine[Any, Any, Model[str, str]] = training_service.train_on_multiple_datasets(self.model, datasets, {},
            {'objective 1': self.objective_function_1, 'objective 2': self.objective_function_2}, None)

        trained_model: Model[str, str] = self.event_loop.run_until_complete(training_routine)

        pre_multi_loop.pre_multi_loop.assert_called()
        post_multi_loop.post_multi_loop.assert_called()
        pre_multi_train_step.pre_multi_train_step.assert_called()
        post_multi_train_step.post_multi_train_step.assert_called()
        pre_loop.pre_loop.assert_called()
        post_loop.post_loop.assert_called()
        pre_epoch.pre_epoch.assert_called()
        post_epoch.post_epoch.assert_called()
        pre_train_step.pre_train.assert_called()
        post_train_step.post_train.assert_called()