from pathlib import Path

from private_eval.regression import evaluate_pretrained_baseline_regression


ROOT = Path(__file__).resolve().parents[1]


def test_pretrained_checkpoint_is_an_explicit_decoder_regression_only():
    result = evaluate_pretrained_baseline_regression(
        official_count=8, shadow_count=8, device="cpu"
    )
    assert result.execution_ok
    assert result.transformer_valid
    assert result.qualifies
    assert result.official_accuracy >= 0.99
    assert result.shadow_accuracy >= 0.99
    assert result.train_seconds == 0.0
