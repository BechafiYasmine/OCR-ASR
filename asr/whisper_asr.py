"""
whisper_asr.py
--------------
Thin wrapper around OpenAI Whisper for transcribing preprocessed audio.

Dependencies:
    pip install -U openai-whisper
    (also requires ffmpeg on PATH)
"""

from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
import whisper


DEFAULT_MODEL_SIZE = "base"  # tiny | base | small | medium | large


@dataclass
class TranscriptionResult:
    text: str
    language: str
    segments: list = field(default_factory=list)  # raw Whisper segments (with timestamps)


@lru_cache(maxsize=4)
def load_model(model_size: str = DEFAULT_MODEL_SIZE):
    """
    Load (and cache) a Whisper model by size. Cached so repeated calls in the
    same process don't reload the weights from disk every time.
    """
    print(f"[whisper_asr] loading Whisper '{model_size}' model...")
    return whisper.load_model(model_size)


def transcribe_array(
    audio: np.ndarray,
    sr: int = 16_000,
    model_size: str = DEFAULT_MODEL_SIZE,
    language: str | None = None,
    fp16: bool = False,
) -> TranscriptionResult:
    """
    Transcribe an in-memory audio array (float32, mono, `sr` Hz — normally the
    output of preprocessing.preprocess_audio).

    Whisper's `transcribe()` expects 16 kHz audio; make sure preprocessing
    already resampled to that rate.
    """
    if sr != 16_000:
        raise ValueError(f"Whisper expects 16 kHz audio, got {sr} Hz. Resample first.")

    model = load_model(model_size)
    result = model.transcribe(audio, language=language, fp16=fp16, verbose=False)

    return TranscriptionResult(
        text=result["text"].strip(),
        language=result.get("language", "unknown"),
        segments=result.get("segments", []),
    )


def transcribe_file(
    path: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    language: str | None = None,
    fp16: bool = False,
) -> TranscriptionResult:
    """Transcribe directly from an audio file path (Whisper handles loading itself)."""
    model = load_model(model_size)
    result = model.transcribe(path, language=language, fp16=fp16, verbose=False)

    return TranscriptionResult(
        text=result["text"].strip(),
        language=result.get("language", "unknown"),
        segments=result.get("segments", []),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python whisper_asr.py <audio_file> [model_size]")
        sys.exit(1)

    audio_path = sys.argv[1]
    size = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL_SIZE

    out = transcribe_file(audio_path, model_size=size)
    print(f"Language: {out.language}")
    print(f"Transcription: {out.text}")
    print(f"Segments: {len(out.segments)}")
