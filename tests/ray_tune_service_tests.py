import asyncio
import unittest
from unittest.mock import MagicMock, Mock, patch
from typing import Any, Coroutine, Iterable, List, Dict, Tuple
from faker import Faker
import random

from src.machine_learning.modeling.model import *
from src.machine_learning.training.batch_training_service import *
from src.machine_learning.evaluation.evaluation_metric import *
from src.machine_learning.training import *
from src.machine_learning.tuning import RayTuneService, TuningService
from ray.tune.schedulers import ASHAScheduler
import ray.tune as tune

class RayTuneServiceTestCase(unittest.TestCase):
    def setUp(self):
        fake = Faker()

        self.event_loop = asyncio.get_event_loop()

    def tearDown(self):
        pass

    def test_tune_valid_training_function_should_run_without_issues(self):
        scheduler = ASHAScheduler(max_t=5, grace_period=1, reduction_factor=2)

        resource_config = {
            "cpu": 1,
            "gpu": 1
        }

        params = {
            "test": tune.grid_search([1, 2, 3])
        }

        tune_config = tune.TuneConfig(
            metric='loss',
            mode='min',
            num_samples=1,
            scheduler=scheduler,
        )

        def training_function(config):
            test = 1

        tuning_service: TuningService = RayTuneService(
            resource_config=resource_config,
            tune_config=tune_config
        )

        self.event_loop.run_until_complete(tuning_service.tune(training_function=training_function, params=params))