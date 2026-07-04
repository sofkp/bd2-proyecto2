import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from backend.src.extractor.sift import SIFTExtractor
from backend.src.index.visual_search import VisualSearchIndex
from backend.src.split.split_image import SplitImage

N_CLUSTERS = 100
MAX_IMAGES = 150
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}

COLOR_H_BINS = 12
COLOR_S_BINS = 4
COLOR_DIM = COLOR_H_BINS + COLOR_S_BINS  

SIFT_W = 0.6
COLOR_W = 0.4


class ImagePipeline:
    def __init__(self) -> None:
        self._splitter = SplitImage(patch_size=224, stride=112)
        self._extractor = SIFTExtractor()
        self._kmeans: KMeans | None = None
        self._index = VisualSearchIndex()
        self.ready = False
        self.indexed_images = 0

    def index_directory(self, images_dir: Path) -> None:
        self.index_directories([images_dir])

    def index_directories(self, images_dirs: list[Path]) -> None:
        # Se respeta el orden de images_dirs: cada carpeta reserva su cupo
        # antes de pasar a la siguiente, para que una carpeta grande (ej.
        # data/full) no desplace por completo a otra (ej. data/samples).
        image_files: list[Path] = []
        for images_dir in images_dirs:
            remaining = MAX_IMAGES - len(image_files)
            if remaining <= 0:
                break
            found = sorted(
                f for f in images_dir.rglob("*") if f.suffix.lower() in SUPPORTED
            )
            image_files.extend(found[:remaining])

        if not image_files:
            return

        per_image: list[dict] = []
        all_desc: list[np.ndarray] = []

        for img_file in image_files:
            try:
                img_array = np.array(Image.open(img_file).convert("RGB"), dtype=np.uint8)
                chunks = self._splitter.split_image(img_array, document_id=img_file.stem)
                patches = [c["content"] for c in chunks]
                desc = self._extractor.extract(patches)
                if desc.shape[0] == 0:
                    continue
                all_desc.append(desc)
                # Prefijo con la carpeta padre: evita colisiones de chunk_id
                # entre fuentes distintas (samples vs. data/full/fashion200k)
                # que puedan compartir el mismo nombre de archivo.
                img_id = f"{img_file.parent.name}_{img_file.stem}"
                url_prefix = "images-full" if "full" in img_file.parts else "images"
                per_image.append({
                    "img_id": img_id,
                    "filename": img_file.name,
                    "image_url": f"/{url_prefix}/{img_file.parent.name}/{img_file.name}"
                        if url_prefix == "images-full" else f"/{url_prefix}/{img_file.name}",
                    "patches": patches,
                    "img_array": img_array,
                })
            except Exception:
                continue

        if not all_desc:
            return

        stacked = np.vstack(all_desc)
        self._kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
        self._kmeans.fit(stacked)

        for item in per_image:
            hist = self._build_histogram(item["patches"], item["img_array"])
            self._index.add_record({
                "chunk_id": item["img_id"],
                "modality": "image",
                "histogram": hist.tolist(),
                "metadata": {
                    "filename": item["filename"],
                    "image_url": item["image_url"],
                    "title": item["img_id"].replace("_", " ").title(),
                },
            })

        self.ready = True
        self.indexed_images = len(per_image)

    def search(self, image_bytes: bytes, k: int = 10) -> list[dict]:
        if not self.ready or self._kmeans is None:
            return []

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img, dtype=np.uint8)
        chunks = self._splitter.split_image(img_array, document_id="query")
        patches = [c["content"] for c in chunks]
        query_hist = self._build_histogram(patches, img_array)

        results = self._index.search(query_hist, k=k)
        return [
            {
                "chunk_id": r.chunk_id,
                "score": round(max(0.0, 1.0 - r.score ** 2 / 2.0), 4),
                "metadata": r.metadata,
            }
            for r in results
        ]

    def index_stats(self) -> dict:
        histograms = self._index._histograms
        n = len(histograms)
        dim = int(next(iter(histograms.values())).shape[0]) if n > 0 else 0
        index_mb = round(n * dim * 4 / (1024 * 1024), 3)
        return {"n_comparisons": n, "vector_dim": dim, "index_mb": index_mb}

    def _build_color_histogram(self, img_array: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        foreground = ~((hsv[:, :, 2] > 230) & (hsv[:, :, 1] < 40))
        if foreground.sum() < 200:
            foreground = np.ones(hsv.shape[:2], dtype=bool)
        h_hist = np.histogram(hsv[:, :, 0][foreground], bins=COLOR_H_BINS, range=(0, 180))[0].astype(np.float32)
        s_hist = np.histogram(hsv[:, :, 1][foreground], bins=COLOR_S_BINS, range=(0, 256))[0].astype(np.float32)
        hist = np.concatenate([h_hist, s_hist])
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    def _build_histogram(self, patches: list[np.ndarray], img_array: np.ndarray | None = None) -> np.ndarray:
        sift_hist = np.zeros(N_CLUSTERS, dtype=np.float32)
        if self._kmeans is not None and patches:
            per_patch = self._extractor.transform(patches)
            for desc in per_patch:
                if desc.shape[0] == 0:
                    continue
                for c in self._kmeans.predict(desc):
                    sift_hist[c] += 1.0
            norm = np.linalg.norm(sift_hist)
            if norm > 0:
                sift_hist /= norm

        if img_array is not None:
            color_hist = self._build_color_histogram(img_array)
            return np.concatenate([sift_hist * SIFT_W, color_hist * COLOR_W])

        return sift_hist


image_pipeline = ImagePipeline()
