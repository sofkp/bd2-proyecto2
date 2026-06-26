from pathlib import Path
import numpy as np
from PIL import Image

PATCH_SIZE = 224
STRIDE = 112
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

class SplitImage:
    def __init__(self, patch_size=PATCH_SIZE, stride=STRIDE, keep_borders=True):
        self.patch_size = patch_size
        self.stride = stride
        self.keep_borders = keep_borders

    def split_file(self, file_path: str, document_id: str = None):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"no existe el archivo: {file_path}")

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"formato no soportado: {file_path.suffix}")

        if document_id is None:
            document_id = self.get_document_id(file_path)

        image = Image.open(file_path).convert("RGB")
        return self.split_image(image, document_id, str(file_path))

    def split_image(self, image, document_id: str, source_path: str = None):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")

        width, height = image.size
        positions_x = self.get_positions(width)
        positions_y = self.get_positions(height)

        chunks = []
        chunk_index = 0

        for y in positions_y:
            for x in positions_x:
                box = (x, y, min(x + self.patch_size, width), min(y + self.patch_size, height))
                patch = image.crop(box)
                original_width, original_height = patch.size

                if original_width < self.patch_size or original_height < self.patch_size:
                    if not self.keep_borders:
                        continue
                    patch = self.pad_patch(patch)

                patch_array = np.array(patch, dtype=np.uint8)

                chunk = {
                    "chunk_id": f"{document_id}_image_{chunk_index}",
                    "document_id": document_id,
                    "modality": "image",
                    "chunk_index": chunk_index,
                    "content": patch_array,
                    "metadata": {
                        "source_path": source_path,
                        "x": x,
                        "y": y,
                        "width": self.patch_size,
                        "height": self.patch_size,
                        "original_width": original_width,
                        "original_height": original_height,
                        "patch_size": self.patch_size,
                        "stride": self.stride,
                        "padded": original_width < self.patch_size or original_height < self.patch_size,
                    },
                }

                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def get_positions(self, size: int):
        if size <= self.patch_size:
            return [0]

        positions = list(range(0, size - self.patch_size + 1, self.stride))

        if not self.keep_borders:
            return positions

        last_start = size - self.patch_size

        if last_start not in positions:
            positions.append(last_start)

        return sorted(set(positions))

    def pad_patch(self, patch):
        new_patch = Image.new("RGB", (self.patch_size, self.patch_size))
        new_patch.paste(patch, (0, 0))
        return new_patch

    def get_document_id(self, file_path):
        return Path(file_path).stem

def print_chunks(chunks):
    print(f"Total patches generados: {len(chunks)}")

    for chunk in chunks[:10]:
        print("\n--------------------------------------------")
        print("chunk_id:", chunk["chunk_id"])
        print("document_id:", chunk["document_id"])
        print("modality:", chunk["modality"])
        print("chunk_index:", chunk["chunk_index"])
        print("x:", chunk["metadata"]["x"])
        print("y:", chunk["metadata"]["y"])
        print("width:", chunk["metadata"]["width"])
        print("height:", chunk["metadata"]["height"])
        print("padded:", chunk["metadata"]["padded"])
        print("original_width:", chunk["metadata"]["original_width"])
        print("original_height:", chunk["metadata"]["original_height"])
        print("shape:", chunk["content"].shape)
        print("dtype:", chunk["content"].dtype)


def test_image():
    curr_dir = Path(__file__).parent
    file_path = curr_dir / "test_image.jpg"

    splitter = SplitImage()

    chunks = splitter.split_file(file_path)

    print_chunks(chunks)


if __name__ == "__main__":
    test_image()