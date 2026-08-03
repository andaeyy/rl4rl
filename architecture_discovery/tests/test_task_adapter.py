import torch

from common.task_adapter import (
    DEFAULT_TASK,
    EOS_ID,
    FIXED_SEQ_LEN,
    PROMPT_LENGTH,
)


def test_answer_only_label_mask_and_eos_are_frozen():
    token_ids, labels = DEFAULT_TASK.encode_example(123, 45)
    assert len(token_ids) == FIXED_SEQ_LEN
    assert labels[:PROMPT_LENGTH] == [-100] * PROMPT_LENGTH
    assert labels[PROMPT_LENGTH:] == token_ids[PROMPT_LENGTH:]
    assert token_ids[-1] == EOS_ID
    assert DEFAULT_TASK.target_text(123, 45) == "86100000000"


def test_collation_preserves_answer_only_mask():
    inputs, labels = DEFAULT_TASK.collate([(0, 0), (9, 1)])
    assert inputs.shape == labels.shape == (2, FIXED_SEQ_LEN)
    assert torch.all(labels[:, :PROMPT_LENGTH] == -100)
    assert torch.equal(labels[:, PROMPT_LENGTH:], inputs[:, PROMPT_LENGTH:])
