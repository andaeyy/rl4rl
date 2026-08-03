import torch

from common.task_adapter import DEFAULT_TASK
from common.training_data import (
    public_development_cases,
    training_batch,
)


def _batch(seed: int):
    return training_batch(
        task=DEFAULT_TASK,
        data_seed=seed,
        optimizer_step=7,
        batch_size=16,
        min_digits=1,
        max_digits=10,
    )


def test_training_batches_are_identical_for_identical_seed_and_step():
    first_inputs, first_labels, first_cases = _batch(101)
    second_inputs, second_labels, second_cases = _batch(101)
    assert first_cases == second_cases
    assert torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_labels, second_labels)


def test_training_batches_change_with_data_seed():
    first_inputs, _, first_cases = _batch(101)
    second_inputs, _, second_cases = _batch(102)
    assert first_cases != second_cases
    assert not torch.equal(first_inputs, second_inputs)


def test_concrete_training_trajectory_excludes_public_development_cases():
    development = public_development_cases(99, 24)
    _, _, cases = training_batch(
        task=DEFAULT_TASK,
        data_seed=88,
        optimizer_step=0,
        batch_size=32,
        min_digits=1,
        max_digits=10,
        excluded_cases=set(development),
    )
    assert set(cases).isdisjoint(development)
