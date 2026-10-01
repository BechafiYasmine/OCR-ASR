"""
timestamps.py
-------------
Extracts and formats word/segment-level timestamps from Whisper's output.

Whisper already returns segment timestamps as part of `transcribe()`, so this
module just reshapes and formats them (e.g. for subtitles or lookup tables).
"""

from dataclasses import dataclass


@dataclass
class Segment:
    index: int
    start: float   # seconds
    end: float     # seconds
    text: str

    def __str__(self) -> str:
        return f"[{format_timestamp(self.start)} -> {format_timestamp(self.end)}] {self.text}"


def format_timestamp(seconds: float) -> str:
    """Convert seconds (float) into an MM:SS timestamp string, e.g. 03.5 -> '00:03'."""
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def extract_segments(whisper_segments: list) -> list[Segment]:
    """
    Convert raw Whisper segment dicts (with 'start', 'end', 'text') into
    clean Segment objects.
    """
    segments = []
    for i, seg in enumerate(whisper_segments):
        segments.append(
            Segment(
                index=i,
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=seg["text"].strip(),
            )
        )
    return segments


def to_srt(segments: list[Segment]) -> str:
    """Render segments as an SRT subtitle file string."""
    def srt_time(t: float) -> str:
        ms = int(round((t - int(t)) * 1000))
        h, rem = divmod(int(t), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for seg in segments:
        lines.append(str(seg.index + 1))
        lines.append(f"{srt_time(seg.start)} --> {srt_time(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def to_json(segments: list[Segment]) -> list[dict]:
    """Render segments as plain JSON-serializable dicts."""
    return [
        {"index": s.index, "start": s.start, "end": s.end, "text": s.text}
        for s in segments
    ]


def find_segment_at(segments: list[Segment], t: float) -> Segment | None:
    """Return the segment that contains a given time `t` (in seconds), if any."""
    for seg in segments:
        if seg.start <= t <= seg.end:
            return seg
    return None


if __name__ == "__main__":
    # quick manual test with fake Whisper-style segments
    fake_segments = [
        {"start": 0.0, "end": 2.4, "text": " Hello there."},
        {"start": 2.4, "end": 6.1, "text": " My internet is very slow."},
    ]
    segs = extract_segments(fake_segments)
    for s in segs:
        print(s)

    print("\n--- SRT ---")
    print(to_srt(segs))
