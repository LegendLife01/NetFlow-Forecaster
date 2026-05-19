#!/usr/bin/env bash
set -euo pipefail

echo "[setup] Updating Ubuntu packages"
sudo apt-get update
sudo apt-get install -y curl ca-certificates python3 python3-pip python3-venv git

if ! command -v docker >/dev/null 2>&1; then
  echo "[setup] Docker CLI not found in WSL."
  echo "Install Docker Desktop on Windows and enable WSL integration for this Ubuntu distro."
  exit 1
fi

echo "[setup] Installing ContainerLab"
bash -c "$(curl -sL https://get.containerlab.dev)"

echo "[setup] Installing Python project dependencies"
python3 -m pip install --user -r requirements.txt

echo ""
echo "[setup] Done. Verify with:"
echo "  docker version"
echo "  containerlab version"
echo "  python3 --version"
