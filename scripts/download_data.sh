#!/bin/bash
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
mkdir -p data/full
export KAGGLEHUB_CACHE="$ROOT/data/full/.kaggle_cache"
[ -d .venv ] && . .venv/bin/activate

ensure_deps() { pip install --quiet kagglehub datasets >/dev/null; }

dl_fashion() {
  ensure_deps
  echo "  fashion-product-images-dataset pesa decenas de GB."
  echo "    Alternativa liviana: paramaggarwal/fashion-product-images-small"
  python3 -c "import kagglehub; print('Descargado en:', kagglehub.dataset_download('paramaggarwal/fashion-product-images-dataset'))"
}
dl_spotify() {
  ensure_deps
  python3 -c "import kagglehub; print('Descargado en:', kagglehub.dataset_download('imuhammad/audio-features-and-lyrics-of-spotify-songs'))"
}
dl_arxiv() {
  ensure_deps
  python3 -c "import datasets; datasets.load_dataset('m-a-p/SciMMIR', cache_dir='data/full/scimmir'); print('SciMMIR descargado en data/full/scimmir')"
}

case "${1:-}" in
  fashion) dl_fashion ;;
  spotify) dl_spotify ;;
  arxiv)   dl_arxiv ;;
  all)     dl_arxiv; dl_spotify; dl_fashion ;;
  *)
    echo "Uso: ./scripts/download_data.sh [arxiv|spotify|fashion|all]"
    echo "  arxiv    SciMMIR (papers de arXiv + figuras) — liviano, recomendado"
    echo "  spotify  audio-features-and-lyrics-of-spotify-songs (~44 MB)"
    echo "  fashion  fashion-product-images-dataset (decenas de GB)"
    echo ""
    echo "Kaggle requiere token una vez: Kaggle → Settings → API → Create New Token"
    echo "Guárdalo en ~/.kaggle/kaggle.json"
    exit 1 ;;
esac
echo " Listo."