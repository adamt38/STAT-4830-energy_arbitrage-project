#!/usr/bin/env bash
# Install uv (if missing), create .venv, and install requirements.
# Run from repo root: bash script/install.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Check for uv
if ! command -v uv &> /dev/null; then
  echo "uv not found. Installing via official Astral installer..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  if ! command -v uv &> /dev/null; then
    echo "uv install may have completed; ensure ~/.local/bin is on PATH and re-run this script."
    exit 1
  fi
  # Ensure future shells find uv: add ~/.local/bin to PATH in shell config
  PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
  for rc in .zshrc .bashrc; do
    rc_path="${HOME}/${rc}"
    if [[ -f "$rc_path" ]] && ! grep -q '.local/bin' "$rc_path" 2>/dev/null; then
      echo "" >> "$rc_path"
      echo "# uv (Python package manager)" >> "$rc_path"
      echo "$PATH_LINE" >> "$rc_path"
      echo "Added $PATH_LINE to $rc_path"
    fi
  done
fi

echo "Using uv: $(uv --version)"

# Create venv and install
uv venv .venv
uv pip install -r requirements.txt

echo ""
echo "--- Next steps ---"
echo "If uv was just installed, either open a new terminal or run:  source ~/.zshrc   (or source ~/.bashrc)"
echo "Activate the virtual environment:"
echo "  source .venv/bin/activate"
echo "Run the main script:"
echo "  python script/gd_1d_torch.py"
echo "Run tests:"
echo "  pytest tests/"
