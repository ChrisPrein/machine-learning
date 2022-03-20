from asyncio import coroutine
import asyncio
from dataclasses import dataclass
from re import S
import unittest
from unittest.mock import Mock, PropertyMock, patch
from torch.utils.data import Dataset
from dataset_handling.dataloader import DataLoader
from typing import Any, Coroutine, List, Dict, Tuple
from faker import Faker
import random

from ..parameter_tuning.custom_model_factory import CustomModelFactory
from ..parameter_tuning.custom_objective_function import CustomObjectiveFunction
from ..evaluation.custom_evaluation_metric import CustomEvaluationMetric
from ..evaluation.abstractions.evaluation_context import EvaluationContext, Prediction, TTarget, TModel, TInput
from ..modeling.abstractions.model import Model

class CustomModelFactoryTestCase(unittest.TestCase):
    def setUp(self):
        fake = Faker()

        self.model_patcher = patch('machine_learning.modeling.abstractions.model', new=Model[float, float])

        self.model: Model[float, float] = self.model_patcher.start()

    def tearDown(self):
        self.model_patcher.stop()

    def test_create_valid_params_should_return_model_instance(self):
        model_factory: CustomModelFactory[Model[float, float]] = CustomModelFactory[Model[float, float]](expression=lambda params: self.model)

        model = model_factory.create({})

        assert not model is None