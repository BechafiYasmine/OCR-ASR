"""
preprocessing.py
-----------------
Cleans and standardizes raw audio before it is sent to an ASR model.

Pipeline:
    raw audio -> mono -> resample (16 kHz) -> trim silence -> reduce noise -> clean audio

Dependencies:
    pip install librosa soundfile numpy noisereduce
"""

from dataclasses import dataclass
import numpy as np
import librosa
import soundfile as sf

try:
    import noisereduce as nr
    _HAS_NOISEREDUCE = True
except ImportError:
    _HAS_NOISEREDUCE = False


TARGET_SR = 16_000  # Whisper expects 16 kHz mono audio


@dataclass
class PreprocessConfig:
    target_sr: int = TARGET_SR
    trim_silence: bool = True
    trim_top_db: float = 25.0       # higher = more aggressive trimming
    reduce_noise: bool = True
    noise_reduce_strength: float = 0.75  # 0..1, how much noise to remove


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file as float32 samples, preserving native sample rate/channels."""
    audio, sr = librosa.load(path, sr=None, mono=False)
    return audio, sr


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse a (channels, samples) array to a single mono channel."""
    if audio.ndim > 1:
        return np.mean(audio, axis=0)
    return audio


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample audio to the target sample rate (16 kHz by default)."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def trim_silence(audio: np.ndarray, top_db: float = 25.0) -> np.ndarray:
    """Remove leading/trailing silence from the signal."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def reduce_noise(audio: np.ndarray, sr: int, strength: float = 0.75) -> np.ndarray:
    """
    Reduce stationary background noise (traffic, hum, chatter) using a
    spectral-gating noise reduction algorithm. Falls back to a no-op if
    the `noisereduce` package isn't installed.
    """
    if not _HAS_NOISEREDUCE:
        return audio
    return nr.reduce_noise(y=audio, sr=sr, prop_decrease=strength)


def preprocess_audio(path: str, config: PreprocessConfig = PreprocessConfig()) -> tuple[np.ndarray, int]:
    """
    Run the full preprocessing pipeline on a single audio file.

    Returns:
        (clean_audio, sample_rate)
    """
    audio, sr = load_audio(path)
    audio = to_mono(audio)
    audio = resample(audio, sr, config.target_sr)
    sr = config.target_sr

    if config.trim_silence:
        audio = trim_silence(audio, top_db=config.trim_top_db)

    if config.reduce_noise:
        audio = reduce_noise(audio, sr, strength=config.noise_reduce_strength)

    # Guard against fully-silent/empty results after trimming
    if audio.size == 0:
        raise ValueError(f"Preprocessing produced empty audio for: {path}")

    return audio.astype(np.float32), sr


def save_audio(audio: np.ndarray, sr: int, out_path: str) -> str:
    """Write processed audio to disk (16-bit PCM WAV)."""
    sf.write(out_path, audio, sr, subtype="PCM_16")
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python preprocessing.py <input_audio> <output_audio>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    clean_audio, sample_rate = preprocess_audio(in_path)
    save_audio(clean_audio, sample_rate, out_path)
    print(f"Saved clean audio -> {out_path} ({sample_rate} Hz, {len(clean_audio) / sample_rate:.2f}s)")
