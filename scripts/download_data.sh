#!/bin/bash
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
mkdir -p data/full
export KAGGLEHUB_CACHE="$ROOT/data/full/.kaggle_cache"
[ -d .venv ] && . .venv/bin/activate

ensure_deps() {
  python3 - <<'PY'
import importlib.util
import subprocess
import sys

missing = [
    package
    for package in ("kagglehub", "datasets")
    if importlib.util.find_spec(package) is None
]
if missing:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--root-user-action=ignore", *missing
    ])
PY
}

dl_agnews() {
  echo "[1/3] Preparando AG News..."
  ensure_deps
  python3 experiments/prepare_text_data.py
  python3 scripts/export_agnews_txt.py
  echo "[1/3] AG News listo."
}

dl_fma_audio() {
  echo "[2/3] Preparando FMA 100K..."
  if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
    if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
      echo "ERROR: FMA requiere credenciales Kaggle."
      echo "Configura KAGGLE_USERNAME y KAGGLE_KEY en .env, o monta ~/.kaggle/kaggle.json."
      exit 1
    fi
  fi
  ensure_deps
  python3 scripts/download_fma_audio.py --materialize --move --clean-cache --write-manifests
  echo "[2/3] FMA listo."
}

dl_fashion200k() {
  echo "[3/3] Preparando Fashion200K..."
  ensure_deps
  python3 experiments/prepare_image_data.py
  echo "[3/3] Fashion200K listo."
}

case "${1:-}" in
  agnews|text) dl_agnews ;;
  fma-audio|fma) dl_fma_audio ;;
  fashion200k|fashion|image) dl_fashion200k ;;
  all) dl_agnews; dl_fma_audio; dl_fashion200k ;;
  *)
    echo "Uso: ./scripts/download_data.sh [agnews|fma-audio|fashion200k|all]"
    echo "  agnews        AG News para texto (1K, 10K, 100K)"
    echo "  fma-audio     FMA dataset 100K WAV files para audio (1K, 10K, 100K)"
    echo "  fashion200k   Marqo/Fashion200K para imagen de ropa (1K, 10K, 100K)"
    echo ""
    echo "Kaggle requiere token una vez para FMA:"
    echo "  Kaggle -> Settings -> API -> Create New Token"
    echo "  Guardar en ~/.kaggle/kaggle.json o usar KAGGLE_USERNAME/KAGGLE_KEY"
    exit 1 ;;
esac
echo " Listo."
