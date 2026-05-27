#!/usr/bin/env bash
# Setup local rapide pour pivot-ai-handball
# Usage : bash scripts/setup_local.sh

set -e

echo "=== pivot-ai-handball : setup local ==="

# Verifier Python 3.11+
if ! command -v python3 &> /dev/null; then
    echo "Erreur : python3 non trouve"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python detecte : $PYTHON_VERSION"

# Creer venv si absent
if [ ! -d ".venv" ]; then
    echo "Creation du venv .venv/"
    python3 -m venv .venv
fi

# Activer venv et installer
echo "Activation venv + installation deps"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Verifier l'install
echo ""
echo "=== Verification ==="
python -c "import pivot_ai; print(f'pivot_ai version {pivot_ai.__version__}')"
python -c "from pivot_ai.config import TerrainConfig; c = TerrainConfig(); print(f'Terrain : {c.longueur_m}x{c.largeur_m}m')"

echo ""
echo "=== Lancement des tests ==="
pytest tests/ -v --tb=short

echo ""
echo "=== Setup OK ==="
echo "Pour activer le venv plus tard : source .venv/bin/activate"
