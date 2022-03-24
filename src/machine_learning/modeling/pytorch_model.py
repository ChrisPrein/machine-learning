from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Dict, Any, Callable
from .abstractions.model import Model, TInput, TTarget
import torch
import torch.nn as nn

class PytorchModel(Model[TInput, TTarget]):
    def __init__(self, pytorch_module: nn.Module, loss_function: nn.Module, optimizer: torch.optim.Optimizer):
        self.inner_module: nn.Module = pytorch_module
        self.loss_function: nn.Module = loss_function
        self.optimizer: torch.optim.Optimizer = optimizer

    def train(self, input: TInput, target: TTarget):
        self.inner_module.train(True)

        self.optimizer.zero_grad()

        output: TTarget = self.inner_module(input)

        loss = self.loss_function(output, target)

        loss.backward()

        self.optimizer.step()
        
        self.inner_module.train(False)


    def train_batch(self, input_batch: List[TInput], target_batch: List[TTarget]):
        self.inner_module.train(True)
        
        self.optimizer.zero_grad()

        output: List[TTarget] = self.inner_module(input)

        loss = self.loss_function(output, target_batch)

        loss.backward()

        self.optimizer.step()
        
        self.inner_module.train(False)

    def predict(self, input: TInput) -> TTarget:
        self.inner_module.train(False)

        return self.inner_module(input)

    def predict_batch(self, input_batch: List[TInput]) -> List[TTarget]:
        self.inner_module.train(False)

        return self.inner_module(input)

    __call__ : Callable[..., Any] = predict