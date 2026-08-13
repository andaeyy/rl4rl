#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
workspace_root=$(CDPATH= cd -- "$project_root/.." && pwd)

if ! command -v git >/dev/null 2>&1; then
    echo "git is required" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

git -C "$workspace_root" submodule update --init --recursive
cd "$project_root"
uv sync --python 3.12 --group modal
.venv/bin/python scripts/openevolve_patch_bundle.py --apply
.venv/bin/python scripts/check_environment.py
.venv/bin/python scripts/validate_configs.py

cat <<'EOF'

Local dependencies and the reviewed OpenEvolve patches are ready.

Configure the shared Modal credentials using Modal's private interactive prompt:

  .venv/bin/modal token set --profile scalingintelligence --activate

Then confirm the selected account without printing the token secret:

  MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main \
    .venv/bin/modal token info

Do not put either Modal token field in this repository or in an .env file.
EOF
