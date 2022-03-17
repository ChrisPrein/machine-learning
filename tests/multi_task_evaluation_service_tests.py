from dataclasses import dataclass
from re import S
import unittest
from unittest.mock import patch
from torch.utils.data import Dataset
from dataset_handling.dataloader import DataLoader
from typing import List
from faker import Faker
import random

@dataclass
class SampleData:
    firstname: str
    lastname: str

class MultiTaskEvaluationServiceTestCase(unittest.TestCase):
    def setUp(self):
        config = {'__getitem__'}
        fake = Faker()

        self.samples: List[SampleData] = [SampleData(fake.first_name(), fake.last_name()) for i in range(10)]

        self.dataset_patcher = patch('torch.utils.data.Dataset')
        self.dataset: Dataset[SampleData] = self.dataset_patcher.start()
        self.dataset.__getitem__.return_value = random.choice(self.samples)
        self.dataset.__len__.return_value = self.samples.__len__()

        self.dataloader: DataLoader[SampleData] = DataLoader[SampleData](self.dataset, batch_size=1, shuffle=True)

    def tearDown(self):
        self.dataset_patcher.stop()

    def test_iteration_should_return_typed_sample(self):
        sample: List[SampleData] = next(iter(self.dataloader))

        assert not sample is None

    def test_iteration_iterate_over_all_samples_twice_should_return_all_samples_twice(self):
        batches: List[List[SampleData]] = [batch for batch in self.dataloader]

        assert len(batches) == len(self.samples)