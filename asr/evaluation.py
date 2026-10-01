"""
evaluation.py
-------------
Computes Word Error Rate (WER) and breaks results down by recording
condition ("clean", "noisy", "fast speech", ...) to support failure
analysis, plus a helper to compare two ASR engines (e.g. Whisper vs Vosk).

Dependencies:
    pip install jiwer
"""

from dataclasses import dataclass, field
import jiwer


# Normalize text before comparing: lowercase, strip punctuation, collapse
# repeated whitespace. This keeps WER focused on actual word errors rather
# than casing/punctuation noise.
_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


@dataclass
class WERResult:
    reference: str
    hypothesis: str
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int


@dataclass
class FileResult:
    file_id: str
    condition: str
    wer_result: WERResult


@dataclass
class EvaluationReport:
    results: list[FileResult] = field(default_factory=list)

    def overall_wer(self) -> float:
        """Aggregate WER computed across ALL files (not the mean of per-file WERs)."""
        refs = [r.wer_result.reference for r in self.results]
        hyps = [r.wer_result.hypothesis for r in self.results]
        return jiwer.wer(refs, hyps, reference_transform=_TRANSFORM, hypothesis_transform=_TRANSFORM)

    def by_condition(self) -> dict:
        """Group WER scores by recording condition, e.g. {'noisy': [0.15, 0.19], ...}."""
        grouped: dict[str, list[float]] = {}
        for r in self.results:
            grouped.setdefault(r.condition, []).append(r.wer_result.wer)
        return grouped

    def condition_summary(self) -> list[tuple[str, float, int]]:
        """Return (condition, avg_wer, num_files) sorted from best to worst."""
        grouped = self.by_condition()
        summary = [
            (cond, sum(scores) / len(scores), len(scores))
            for cond, scores in grouped.items()
        ]
        return sorted(summary, key=lambda x: x[1])

    def worst_files(self, n: int = 5) -> list[FileResult]:
        """Return the n files with the highest WER — useful for spot-checking failures."""
        return sorted(self.results, key=lambda r: r.wer_result.wer, reverse=True)[:n]


def compute_wer(reference: str, hypothesis: str) -> WERResult:
    """Compute WER plus the substitution/deletion/insertion breakdown for one pair."""
    output = jiwer.process_words(
        reference, hypothesis,
        reference_transform=_TRANSFORM,
        hypothesis_transform=_TRANSFORM,
    )
    return WERResult(
        reference=reference,
        hypothesis=hypothesis,
        wer=output.wer,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        hits=output.hits,
    )


def evaluate_dataset(samples: list[dict]) -> EvaluationReport:
    """
    samples: list of dicts like
        {"file_id": "audio_01", "condition": "clean",
         "reference": "...", "hypothesis": "..."}
    """
    report = EvaluationReport()
    for s in samples:
        wer_result = compute_wer(s["reference"], s["hypothesis"])
        report.results.append(FileResult(
            file_id=s["file_id"],
            condition=s.get("condition", "unspecified"),
            wer_result=wer_result,
        ))
    return report


def compare_engines(samples: list[dict]) -> dict:
    """
    Compare two ASR engines (e.g. Whisper vs Vosk) on the same reference set.

    samples: list of dicts like
        {"file_id": "audio_01", "reference": "...",
         "whisper_hyp": "...", "vosk_hyp": "..."}

    Returns overall WER per engine.
    """
    refs = [s["reference"] for s in samples]
    whisper_hyps = [s["whisper_hyp"] for s in samples]
    vosk_hyps = [s["vosk_hyp"] for s in samples]

    return {
        "whisper_wer": jiwer.wer(refs, whisper_hyps, reference_transform=_TRANSFORM, hypothesis_transform=_TRANSFORM),
        "vosk_wer": jiwer.wer(refs, vosk_hyps, reference_transform=_TRANSFORM, hypothesis_transform=_TRANSFORM),
    }


def print_report(report: EvaluationReport) -> None:
    print(f"Overall WER: {report.overall_wer() * 100:.1f}%\n")

    print("By condition:")
    for cond, avg_wer, n in report.condition_summary():
        print(f"  {cond:<20} {avg_wer * 100:5.1f}%   (n={n})")

    print("\nWorst-performing files:")
    for r in report.worst_files():
        w = r.wer_result
        print(f"  {r.file_id:<12} [{r.condition:<15}] WER={w.wer*100:5.1f}%  "
              f"(sub={w.substitutions}, del={w.deletions}, ins={w.insertions})")


if __name__ == "__main__":
    demo_samples = [
        {"file_id": "audio_01", "condition": "clean",
         "reference": "My internet is very slow.", "hypothesis": "My internet is very slow."},
        {"file_id": "audio_02", "condition": "background_noise",
         "reference": "My internet is very slow.", "hypothesis": "My internet is really slow."},
        {"file_id": "audio_05", "condition": "low_quality",
         "reference": "I have a problem with my internet connection.",
         "hypothesis": "I have a problem my internet connection."},
    ]
    report = evaluate_dataset(demo_samples)
    print_report(report)
