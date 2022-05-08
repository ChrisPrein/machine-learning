from abc import ABC, abstractmethod
from ast import Call
from typing import Optional, TypeVar, Generic, List, Dict, Any, Callable, Union
from .abstractions.model import Model, TInput, TTarget
import torch
import torch.nn as nn

from ..exceptions.no_training_in_progress import NoTrainingInProgressException

class PytorchModel(Model[TInput, TTarget]):
    def __init__(self, pytorch_module: nn.Module, loss_function: nn.Module, optimizer: Union[torch.optim.Optimizer, Callable[[], torch.optim.Optimizer]], scheduler: Optional[Union[torch.optim.lr_scheduler._LRScheduler, Callable[[], torch.optim.lr_scheduler._LRScheduler]]] = None):
        self.inner_module: nn.Module = pytorch_module
        self.loss_function: nn.Module = loss_function

        self.optimizer_factory: Optional[Callable[[], torch.optim.Optimizer]] = optimizer if not isinstance(optimizer, torch.optim.Optimizer) else None
        self.scheduler_factory: Optional[Callable[[], torch.optim.lr_scheduler._LRScheduler]] = scheduler if not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler) else None

        self.optimizer: Optional[torch.optim.Optimizer] = optimizer if isinstance(optimizer, torch.optim.Optimizer) else None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = scheduler if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler) else None

        self._is_training_in_progress: bool = False

    def start_training(self):
        self._is_training_in_progress = True

        if not self.optimizer_factory is None:
            self.optimizer = self.optimizer_factory()

        if not self.scheduler_factory is None:
            self.scheduler = self.scheduler_factory()

    def end_training(self):
        self._is_training_in_progress = False

    def train(self, input: TInput, target: TTarget):
        if not self._is_training_in_progress:
            raise NoTrainingInProgressException("Training has to be started before calling train.")

        self.inner_module.train(True)

        self.optimizer.zero_grad()

        output: TTarget = self.inner_module(input)

        loss = self.loss_function(output, target)

        loss.backward()

        self.optimizer.step()

        if not self.scheduler is None:
            self.scheduler.step()
        
        self.inner_module.train(False)


    def train_batch(self, input_batch: List[TInput], target_batch: List[TTarget]):
        if not self._is_training_in_progress:
            raise NoTrainingInProgressException("Training has to be started before calling train_batch.")

        self.inner_module.train(True)
        
        self.optimizer.zero_grad()

        output: List[TTarget] = self.inner_module(input_batch)

        loss = self.loss_function(output, target_batch)

        loss.backward()

        self.optimizer.step()

        if not self.scheduler is None:
            self.scheduler.step()
        
        self.inner_module.train(False)

    def predict(self, input: TInput) -> TTarget:
        self.inner_module.train(False)

        return self.inner_module(input)

    def predict_batch(self, input_batch: List[TInput]) -> List[TTarget]:
        self.inner_module.train(False)

        return self.inner_module(input_batch)

    __call__ : Callable[..., Any] = predict