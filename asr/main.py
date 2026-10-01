"""
main.py
-------
Orchestrates the full ASR pipeline end to end:

    audio -> preprocessing -> Whisper -> transcription + timestamps
          -> WER evaluation -> failure analysis

Usage:
    # Transcribe a single file
    python main.py transcribe path/to/audio.wav

    # Evaluate a batch of files against reference transcripts
    python main.py evaluate data/manifest.json

manifest.json format:
    [
      {"file_id": "audio_01", "path": "data/audio_01.wav",
       "condition": "clean", "reference": "My internet is very slow."},
      ...
    ]
"""

import argparse
import json
import os
import sys

from preprocessing import preprocess_audio, PreprocessConfig
from whisper_asr import transcribe_array, DEFAULT_MODEL_SIZE
from timestamps import extract_segments
from evaluation import evaluate_dataset, print_report


def run_transcription(audio_path: str, model_size: str = DEFAULT_MODEL_SIZE) -> dict:
    """Run one file through the full pipeline: preprocess -> Whisper -> timestamps."""
    print(f"[main] preprocessing: {audio_path}")
    clean_audio, sr = preprocess_audio(audio_path, PreprocessConfig())

    print(f"[main] transcribing with Whisper ({model_size})...")
    result = transcribe_array(clean_audio, sr=sr, model_size=model_size)

    segments = extract_segments(result.segments)

    return {
        "file": audio_path,
        "language": result.language,
        "text": result.text,
        "segments": [str(s) for s in segments],
    }


def run_batch_evaluation(manifest_path: str, model_size: str = DEFAULT_MODEL_SIZE):
    """
    Read a manifest of {file_id, path, condition, reference}, transcribe every
    file with Whisper, then compute WER and print a failure-analysis report.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = []
    for entry in manifest:
        print(f"[main] processing {entry['file_id']}...")
        clean_audio, sr = preprocess_audio(entry["path"], PreprocessConfig())
        result = transcribe_array(clean_audio, sr=sr, model_size=model_size)

        samples.append({
            "file_id": entry["file_id"],
            "condition": entry.get("condition", "unspecified"),
            "reference": entry["reference"],
            "hypothesis": result.text,
        })

    report = evaluate_dataset(samples)
    print()
    print_report(report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASR pipeline: preprocessing -> Whisper -> WER evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_transcribe = sub.add_parser("transcribe", help="Transcribe a single audio file")
    p_transcribe.add_argument("audio_path", help="Path to an audio file")
    p_transcribe.add_argument("--model", default=DEFAULT_MODEL_SIZE, help="Whisper model size")

    p_evaluate = sub.add_parser("evaluate", help="Run batch transcription + WER evaluation")
    p_evaluate.add_argument("manifest_path", help="Path to a JSON manifest (see module docstring)")
    p_evaluate.add_argument("--model", default=DEFAULT_MODEL_SIZE, help="Whisper model size")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "transcribe":
        if not os.path.exists(args.audio_path):
            print(f"File not found: {args.audio_path}", file=sys.stderr)
            sys.exit(1)
        result = run_transcription(args.audio_path, model_size=args.model)
        print("\n--- Transcription ---")
        print(result["text"])
        print("\n--- Segments ---")
        for seg in result["segments"]:
            print(seg)

    elif args.command == "evaluate":
        if not os.path.exists(args.manifest_path):
            print(f"Manifest not found: {args.manifest_path}", file=sys.stderr)
            sys.exit(1)
        run_batch_evaluation(args.manifest_path, model_size=args.model)


if __name__ == "__main__":
    main()
