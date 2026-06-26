from pathlib import Path
import numpy as np
import librosa

SAMPLE_RATE = 22050
WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5
SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

class SplitAudio:
    def __init__(self, sample_rate=SAMPLE_RATE, window_seconds=WINDOW_SECONDS, hop_seconds=HOP_SECONDS, keep_last_window=True):
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.keep_last_window = keep_last_window

        self.window_size = int(self.sample_rate * self.window_seconds)
        self.hop_size = int(self.sample_rate * self.hop_seconds)

        if self.window_size <= 0:
            raise ValueError("window_size debe ser mayor a 0")

        if self.hop_size <= 0:
            raise ValueError("hop_size debe ser mayor a 0")

    def split_file(self, file_path: str, document_id: str = None):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"no existe el archivo: {file_path}")

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"formato no soportado: {file_path.suffix}")

        if document_id is None:
            document_id = self.get_document_id(file_path)

        audio, sr = librosa.load(path=str(file_path), sr=self.sample_rate, mono=True)

        return self.split_audio(audio, document_id, str(file_path))

    def split_audio(self, audio: np.ndarray, document_id: str, source_path: str = None):
        audio = self.clean_audio(audio)

        chunks = []
        chunk_index = 0

        positions = self.get_positions(len(audio))

        for start in positions:
            end = start + self.window_size
            window = audio[start:end]
            original_size = len(window)

            if original_size < self.window_size:
                if not self.keep_last_window:
                    continue
                window = self.pad_window(window)

            window = window.astype(np.float32)

            chunk = {
                "chunk_id": f"{document_id}_audio_{chunk_index}",
                "document_id": document_id,
                "doc_id": document_id,
                "modality": "audio",
                "chunk_index": chunk_index,
                "content": window,
                "metadata": {
                    "source_path": source_path,
                    "sample_rate": self.sample_rate,
                    "start_sample": start,
                    "end_sample": min(end, len(audio)),
                    "start_second": start / self.sample_rate,
                    "end_second": min(end, len(audio)) / self.sample_rate,
                    "window_size": self.window_size,
                    "hop_size": self.hop_size,
                    "window_seconds": self.window_seconds,
                    "hop_seconds": self.hop_seconds,
                    "original_size": original_size,
                    "padded": original_size < self.window_size,
                },
            }

            chunks.append(chunk)
            chunk_index += 1

        return chunks

    def clean_audio(self, audio):
        if audio is None:
            return np.array([], dtype=np.float32)

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        return audio

    def get_positions(self, audio_size: int):
        if audio_size <= self.window_size:
            return [0]

        positions = list(range(0, audio_size - self.window_size + 1, self.hop_size))

        if self.keep_last_window:
            last_start = audio_size - self.window_size

            if last_start not in positions:
                positions.append(last_start)

        return sorted(set(positions))

    def pad_window(self, window):
        padded = np.zeros(self.window_size, dtype=np.float32)
        padded[:len(window)] = window
        return padded

    def get_document_id(self, file_path):
        return Path(file_path).stem


def print_chunks(chunks):
    print(f"Total ventanas generadas: {len(chunks)}")

    for chunk in chunks[:10]:
        print("\n--------------------------------------------")
        print("chunk_id:", chunk["chunk_id"])
        print("document_id:", chunk["document_id"])
        print("modality:", chunk["modality"])
        print("chunk_index:", chunk["chunk_index"])
        print("start_second:", round(chunk["metadata"]["start_second"], 3))
        print("end_second:", round(chunk["metadata"]["end_second"], 3))
        print("window_seconds:", chunk["metadata"]["window_seconds"])
        print("hop_seconds:", chunk["metadata"]["hop_seconds"])
        print("padded:", chunk["metadata"]["padded"])
        print("shape:", chunk["content"].shape)
        print("dtype:", chunk["content"].dtype)


def test_audio():
    curr_dir = Path(__file__).parent
    file_path = curr_dir / "test-audio.mp3"

    splitter = SplitAudio()

    chunks = splitter.split_file(file_path)

    print_chunks(chunks)


if __name__ == "__main__":
    test_audio()