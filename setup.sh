#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Allow overrides: PYTHON_BIN=python3.11 VENV_DIR=.venv ./setup.sh
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_DEV="${INSTALL_DEV:-1}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: could not find python executable (tried python3 and python)." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [ "$INSTALL_DEV" = "1" ]; then
  python -m pip install -e ".[dev]"
else
  python -m pip install -e .
fi

echo "Setup complete."
echo "Virtual environment: $ROOT_DIR/$VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
