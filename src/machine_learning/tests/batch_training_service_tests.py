import asyncio
import unittest
from unittest.mock import MagicMock, Mock, patch
from torch.utils.data import Dataset
from typing import Any, Coroutine, List, Dict, Tuple
from faker import Faker
import random
from ..modeling.abstractions.model import Model, TInput, TTarget
from ..training.batch_training_service import BatchTrainingService
from ..training.abstractions.objective_function import ObjectiveFunction

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