"""Evaluator-owned Phase-1 addition task semantics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


TOKENS = tuple(str(i) for i in range(10)) + (
    "+",
    "=",
    "<pad>",
    "<bos>",
    "<eos>",
)
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}
ID_TO_TOKEN = {index: token for token, index in TOKEN_TO_ID.items()}
PAD_ID = TOKEN_TO_ID["<pad>"]
BOS_ID = TOKEN_TO_ID["<bos>"]
EOS_ID = TOKEN_TO_ID["<eos>"]
VOCAB_SIZE = len(TOKENS)
NUM_DIGITS = 10
OUT_DIGITS = 11
PROMPT_LENGTH = 1 + NUM_DIGITS + 1 + NUM_DIGITS + 1
FIXED_SEQ_LEN = PROMPT_LENGTH + OUT_DIGITS + 1
TASK_ADAPTER_VERSION = "fixed_addition_v1"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FixedAdditionTask:
    """Declarative frozen task adapter.

    Tokenization mutations are intentionally outside the Phase-1 contract.
    """

    version: str = TASK_ADAPTER_VERSION
    vocabulary: tuple[str, ...] = TOKENS
    operand_digits: int = NUM_DIGITS
    output_digits: int = OUT_DIGITS
    reverse_output: bool = True

    @property
    def config_hash(self) -> str:
        return _sha256_text(
            "|".join(
                (
                    self.version,
                    ",".join(self.vocabulary),
                    str(self.operand_digits),
                    str(self.output_digits),
                    str(self.reverse_output),
                )
            )
        )

    def prompt_text(self, a: int, b: int) -> str:
        self.validate_operands(a, b)
        return f"{a:0{self.operand_digits}d}+{b:0{self.operand_digits}d}="

    def target_text(self, a: int, b: int) -> str:
        self.validate_operands(a, b)
        normal = f"{a + b:0{self.output_digits}d}"
        return normal[::-1] if self.reverse_output else normal

    def validate_operands(self, a: int, b: int) -> None:
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("operands must be integers")
        upper = 10**self.operand_digits - 1
        if min(a, b) < 0 or max(a, b) > upper:
            raise ValueError("operands fall outside the fixed Phase-1 range")

    def encode_text(self, text: str) -> list[int]:
        return [TOKEN_TO_ID[character] for character in text]

    def encode_prompt(self, a: int, b: int) -> list[int]:
        return [BOS_ID] + self.encode_text(self.prompt_text(a, b))

    def encode_example(self, a: int, b: int) -> tuple[list[int], list[int]]:
        token_ids = (
            self.encode_prompt(a, b)
            + self.encode_text(self.target_text(a, b))
            + [EOS_ID]
        )
        if len(token_ids) != FIXED_SEQ_LEN:
            raise AssertionError(
                f"task adapter emitted {len(token_ids)} tokens, expected {FIXED_SEQ_LEN}"
            )
        labels = [-100] * PROMPT_LENGTH + token_ids[PROMPT_LENGTH:]
        return token_ids, labels

    def collate(
        self, cases: Iterable[tuple[int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode_example(a, b) for a, b in cases]
        return (
            torch.tensor([item[0] for item in encoded], dtype=torch.long),
            torch.tensor([item[1] for item in encoded], dtype=torch.long),
        )

    def teacher_forced_loss(
        self, model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        logits = model(input_ids)
        if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
            raise ValueError(
                "candidate forward must return [batch, sequence, vocabulary] logits"
            )
        if logits.shape[-1] != len(self.vocabulary):
            raise ValueError(
                f"candidate emitted vocabulary size {logits.shape[-1]}, "
                f"expected {len(self.vocabulary)}"
            )
        return F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def generate(
        self, model: torch.nn.Module, prompt_ids: torch.Tensor
    ) -> torch.Tensor:
        generated = prompt_ids
        for _ in range(self.output_digits + 1):
            max_sequence = int(getattr(model, "max_seq_len", FIXED_SEQ_LEN))
            logits = model(generated[:, -max_sequence:])
            if logits.ndim != 3 or logits.shape[-1] != len(self.vocabulary):
                raise ValueError("candidate forward emitted invalid logits")
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
        return generated

    def decode_generated(self, generated_ids: list[int]) -> int:
        answer = generated_ids[: self.output_digits]
        digits: list[str] = []
        for token_id in answer:
            if token_id == EOS_ID:
                break
            token = ID_TO_TOKEN.get(token_id, "")
            if token.isdigit():
                digits.append(token)
        text = "".join(digits)
        if not text:
            return 0
        normal = text[::-1] if self.reverse_output else text
        return int(normal)

    @torch.no_grad()
    def exact_match(
        self,
        model: torch.nn.Module,
        cases: list[tuple[int, int]],
        *,
        device: torch.device,
        batch_size: int,
        failure_limit: int = 20,
    ) -> tuple[float, list[dict[str, int]]]:
        was_training = model.training
        model.eval()
        correct = 0
        failures: list[dict[str, int]] = []
        for offset in range(0, len(cases), batch_size):
            batch = cases[offset : offset + batch_size]
            prompts = torch.tensor(
                [self.encode_prompt(a, b) for a, b in batch],
                dtype=torch.long,
                device=device,
            )
            generated = self.generate(model, prompts)
            prompt_length = prompts.shape[1]
            for row, (a, b) in enumerate(batch):
                completion = generated[
                    row, prompt_length : prompt_length + self.output_digits + 1
                ].detach().cpu().tolist()
                expected_tokens = self.encode_text(self.target_text(a, b)) + [EOS_ID]
                observed = self.decode_generated(completion)
                expected = a + b
                if completion == expected_tokens:
                    correct += 1
                elif len(failures) < failure_limit:
                    failures.append(
                        {
                            "a": a,
                            "b": b,
                            "expected": expected,
                            "observed": observed,
                        }
                    )
        model.train(was_training)
        return (correct / len(cases) if cases else 0.0), failures


DEFAULT_TASK = FixedAdditionTask()
