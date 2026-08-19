"""Shared bounded contract for convenient engineering evolution runs."""

from __future__ import annotations

import re
from dataclasses import dataclass

EVOLUTION_ACTION = "evolve"
EVOLUTION_FUNCTION_NAME = "evolution_run"
EVOLUTION_MAX_ITERATIONS = 345
EVOLUTION_INPUT_BYTES_PER_REQUEST = 1_048_576
EVOLUTION_COMPLETION_TOKENS_PER_REQUEST = 16_384
EVOLUTION_PROVIDER_TIMEOUT_SECONDS = 180
EVOLUTION_TRAINING_TIMEOUT_SECONDS = 60
EVOLUTION_FINALIZATION_RESERVE_SECONDS = 300
EVOLUTION_IMAGE_BUILD_TIMEOUT_SECONDS = 600
EVOLUTION_CLI_RESERVE_SECONDS = 300

EVOLUTION_HARNESSES = (
    "greedy_autoresearch",
    "semantic_autoresearch",
    "openevolve_generic",
    "openevolve_semantic",
)
EVOLUTION_ENGINE_ALIASES = {
    "autoresearch": "greedy_autoresearch",
    "semantic-autoresearch": "semantic_autoresearch",
    "openevolve": "openevolve_generic",
    "semantic-openevolve": "openevolve_semantic",
}
_SPEC = re.compile(
    rf"\A({'|'.join(re.escape(item) for item in EVOLUTION_HARNESSES)})-n([1-9][0-9]*)\Z"
)


@dataclass(frozen=True, slots=True)
class EvolutionRunSpec:
    harness: str
    iterations: int

    def __post_init__(self) -> None:
        if self.harness not in EVOLUTION_HARNESSES:
            raise ValueError("evolution harness is unsupported")
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or not 1 <= self.iterations <= EVOLUTION_MAX_ITERATIONS
        ):
            raise ValueError(
                f"iterations must be between 1 and {EVOLUTION_MAX_ITERATIONS}"
            )

    @property
    def token(self) -> str:
        return f"{self.harness}-n{self.iterations}"

    @property
    def controller_timeout_seconds(self) -> int:
        # One initial smoke evaluation, then one provider request and one smoke
        # evaluation per opportunity. Reserve at least one minute plus nine
        # seconds per opportunity for controller bookkeeping.
        return (
            EVOLUTION_TRAINING_TIMEOUT_SECONDS
            + self.iterations
            * (
                EVOLUTION_PROVIDER_TIMEOUT_SECONDS
                + EVOLUTION_TRAINING_TIMEOUT_SECONDS
            )
            + max(60, 9 * self.iterations)
        )

    @property
    def function_timeout_seconds(self) -> int:
        return (
            self.controller_timeout_seconds
            + EVOLUTION_FINALIZATION_RESERVE_SECONDS
        )

    @property
    def outer_cli_timeout_seconds(self) -> int:
        return (
            EVOLUTION_IMAGE_BUILD_TIMEOUT_SECONDS
            + self.function_timeout_seconds
            + EVOLUTION_CLI_RESERVE_SECONDS
        )

    @classmethod
    def parse(cls, token: str) -> "EvolutionRunSpec":
        if not isinstance(token, str):
            raise TypeError("evolution spec must be text")
        match = _SPEC.fullmatch(token)
        if match is None:
            raise ValueError("evolution spec must be <harness>-n<iterations>")
        return cls(match.group(1), int(match.group(2)))

    @classmethod
    def from_cli(cls, engine: str, iterations: int) -> "EvolutionRunSpec":
        try:
            harness = EVOLUTION_ENGINE_ALIASES[engine]
        except KeyError as error:
            raise ValueError("evolution engine alias is unsupported") from error
        return cls(harness, iterations)
