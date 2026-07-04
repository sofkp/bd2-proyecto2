"""
Genera clips de audio sintéticos cortos para data/samples/audio/, de modo
que el pipeline de audio (MFCC + Codebook) tenga datos de ejemplo listos
para indexar sin depender de la descarga de GTZAN (data/full/audio).

No requiere numpy/scipy: usa únicamente los módulos wave y math/random
de la librería estándar.
"""
import math
import random
import wave
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "samples" / "audio"
SAMPLE_RATE = 22050
DURATION_SECONDS = 6.0
AMPLITUDE = 0.5


def _write_wav(path: Path, samples: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(SAMPLE_RATE)
        frames = bytearray()
        for s in samples:
            value = int(max(-1.0, min(1.0, s)) * 32767)
            frames += value.to_bytes(2, byteorder="little", signed=True)
        wav_file.writeframes(bytes(frames))


def _tone(freq: float, n: int, wobble: float = 0.0) -> list[float]:
    out = []
    for i in range(n):
        t = i / SAMPLE_RATE
        f = freq + wobble * math.sin(2 * math.pi * 0.5 * t)
        out.append(AMPLITUDE * math.sin(2 * math.pi * f * t))
    return out


def _noise(n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [AMPLITUDE * (rng.random() * 2 - 1) for _ in range(n)]


def main() -> None:
    n = int(SAMPLE_RATE * DURATION_SECONDS)

    tracks = {
        "tono_grave": [
            ("clip_a", _tone(110.0, n)),
            ("clip_b", _tone(146.83, n, wobble=3.0)),
        ],
        "tono_agudo": [
            ("clip_a", _tone(880.0, n)),
            ("clip_b", _tone(987.77, n, wobble=8.0)),
        ],
        "ruido": [
            ("clip_a", _noise(n, seed=1)),
            ("clip_b", _noise(n, seed=2)),
        ],
    }

    for genre, clips in tracks.items():
        for name, samples in clips:
            # chunk_id se deriva solo del nombre de archivo (sin carpeta), así
            # que el nombre debe ser único entre TODAS las carpetas de género.
            out_path = OUTPUT_DIR / genre / f"{genre}_{name}.wav"
            _write_wav(out_path, samples)
            print(f"  {out_path.relative_to(OUTPUT_DIR.parent.parent.parent)}")

    print("Listo: audios de ejemplo generados en data/samples/audio/")


if __name__ == "__main__":
    main()
