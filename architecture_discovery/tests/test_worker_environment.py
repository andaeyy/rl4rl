from pathlib import Path

from common.training_client import build_worker_environment


ROOT = Path(__file__).resolve().parents[1]


def test_worker_environment_is_allowlisted_and_excludes_credentials():
    secrets = {
        "DISCOVERY_API_KEY": "discovery-secret",
        "OPENAI_API_KEY": "openai-secret",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "GOOGLE_API_KEY": "google-secret",
        "GEMINI_API_KEY": "gemini-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "GITHUB_TOKEN": "github-secret",
        "HF_TOKEN": "hf-secret",
        "SUPER_SECRET_TOKEN": "other-secret",
        "DISCOVERY_SHADOW_SEED": "sealed-seed",
        "PYTHONPATH": "/untrusted",
        "PYTHONSTARTUP": "/untrusted/start.py",
        "LANG": "en_US.UTF-8",
    }
    environment = build_worker_environment(
        requested_device="cpu",
        allow_cpu_for_tests=True,
        model_seed=1,
        parent_environment=secrets,
    )
    assert environment["LANG"] == "en_US.UTF-8"
    assert environment["DISCOVERY_TRAIN_DEVICE"] == "cpu"
    assert environment["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    for key, value in secrets.items():
        if key == "LANG":
            continue
        assert key not in environment
        assert value not in environment.values()


def test_training_path_has_no_provider_sdk_or_paid_call():
    for name in (
        "trainer.py",
        "training_client.py",
        "training_worker.py",
        "training_data.py",
    ):
        source = (ROOT / "common" / name).read_text().lower()
        assert "from openai" not in source
        assert "import openai" not in source
        assert "chat.completions" not in source
