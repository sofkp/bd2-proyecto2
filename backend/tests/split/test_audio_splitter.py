import numpy as np
from backend.src.split.split_audio import SplitAudio

def test_audio_splitter_creates_overlapped_windows_with_timestamps():
    audio = np.arange(10, dtype=np.float32)
    splitter = SplitAudio(sample_rate=10, window_seconds=0.4, hop_seconds=0.2)

    chunks = splitter.split_audio(audio, document_id="aud1")

    assert len(chunks) == 4
    assert [chunk["metadata"]["start_sample"] for chunk in chunks] == [0, 2, 4, 6]

    first = chunks[0]
    assert first["chunk_id"] == "aud1_audio_0"
    assert first["chunk_index"] == 0
    assert first["modality"] == "audio"
    assert first["content"].shape == (4,)
    assert first["metadata"]["sample_rate"] == 10
    assert first["metadata"]["start_second"] == 0.0
    assert first["metadata"]["end_second"] == 0.4
