import numpy as np
from backend.src.split.split_image import SplitImage


def test_image_splitter_creates_configurable_patches_with_positions():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    splitter = SplitImage(patch_size=2, stride=2)

    chunks = splitter.split_image(image, document_id="img1")

    positions = [(chunk["metadata"]["x"], chunk["metadata"]["y"]) for chunk in chunks]
    assert positions == [(0, 0), (2, 0), (0, 2), (2, 2)]

    assert len(chunks) == 4
    assert chunks[0]["chunk_id"] == "img1_image_0"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["modality"] == "image"
    assert chunks[0]["content"].shape == (2, 2, 3)
    assert chunks[0]["metadata"]["patch_size"] == 2
    assert chunks[0]["metadata"]["stride"] == 2
