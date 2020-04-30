#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mlbench_core.controlflow.pytorch` package."""

import pytest

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random

from mlbench_core.utils import Tracker
from mlbench_core.controlflow.pytorch import MLBenchTrainer
from mlbench_core.evaluation.pytorch.metrics import TopKAccuracy
from mlbench_core.lr_scheduler.pytorch import multistep_learning_rates_with_warmup


@pytest.fixture
def model():
    return nn.Linear(1, 2)


@pytest.fixture
def optimizer(model):
    return optim.SGD(model.parameters(), lr=0.1)


@pytest.fixture
def loss_function():
    return nn.CrossEntropyLoss()


@pytest.fixture
def metrics():
    return [TopKAccuracy(topk=1)]


@pytest.fixture
def scheduler(optimizer):
    return multistep_learning_rates_with_warmup(
        optimizer,
        1,
        0.1,
        0.1,
        [5, 10],
        warmup_duration=2,
        warmup_linear_scaling=False,
        warmup_lr=0.2,
    )


def test_instantiation(mocker, model, optimizer, loss_function, metrics, scheduler):
    mocker.patch("mlbench_core.controlflow.pytorch.controlflow.dist")

    tv = MLBenchTrainer(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        metrics=metrics,
        scheduler=scheduler,
        rank=0
    )

    assert tv is not None


def test_training(mocker, model, optimizer, loss_function, metrics, scheduler):
    mocker.patch("mlbench_core.controlflow.pytorch.controlflow.dist")
    mocker.patch("mlbench_core.utils.pytorch.distributed.dist")
    mocker.patch("mlbench_core.utils.tracker.LogMetrics")

    batch_size = 2
    tracker = Tracker(metrics, 1, rank=0)
    tv = MLBenchTrainer(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        metrics=metrics,
        scheduler=scheduler,
        rank=0,
        tracker=tracker
    )

    train_set = [random.random() for _ in range(100)]
    train_set = [
        (
            torch.FloatTensor([n * 50 - 25]),
            1 if (n > 0.5) != (random.random() < 0.1) else 0,
        )
        for n in train_set
    ]

    test_set = [random.random() for _ in range(10)]
    test_set = [
        (
            torch.FloatTensor([n * 50 - 25]),
            1 if (n > 0.5) != (random.random() < 0.1) else 0,
        )
        for n in test_set
    ]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    n_epochs = 10
    for e in range(n_epochs):
        tv.train_round(train_loader, dtype="fp32")
        tv.validation_round(test_loader, dtype="fp32", )

    assert tv.tracker.current_epoch == 10
    assert tv.tracker.best_epoch > -1
    assert tv.tracker.best_metric_value > 50.0
