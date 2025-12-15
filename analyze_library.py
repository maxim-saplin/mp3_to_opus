from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


SUPPORTED_AUDIO_EXTS = {
    ".mp3",
    ".flac",
    ".wav",
    ".m4a",
    ".ogg",
    ".wma",
    ".aac",
    ".alac",
    ".aiff",
    ".ape",
    ".dsf",
}


@dataclass
class ExtStats:
    count: int = 0
    size_bytes: int = 0


def _run_ffprobe_json(path: Path) -> dict[str, Any] | None:
    """
    Returns ffprobe JSON dict, or None if ffprobe fails.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format=duration,bit_rate",
        "-show_format",
        str(path),
    ]
    try:
        # Use text=False so Windows paths/unicode are passed safely.
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found on PATH. Install FFmpeg and ensure ffprobe is on PATH.")

    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None


def _get_format_number(info: dict[str, Any] | None, key: str) -> float | None:
    if not info:
        return None
    fmt = info.get("format") or {}
    val = fmt.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _iter_files(root: Path) -> Iterable[Path]:
    # rglob on Windows is Unicode-safe in Python 3.
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _human_bytes(num_bytes: int) -> str:
    if num_bytes < 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    n = float(num_bytes)
    for u in units:
        if n < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(n)} {u}"
            return f"{n:.2f} {u}"
        n /= 1024.0
    return f"{num_bytes} B"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze an audio library: extensions, MP3 bitrate buckets, and 128kbps size estimate.")
    parser.add_argument("--src", required=True, help="Source folder to analyze (recursive).")
    parser.add_argument("--target-bitrate-kbps", type=int, default=128, help="Bitrate used for size estimate (default: 128).")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel ffprobe workers (default: cpu_count-1). Higher is faster but may stress disk.",
    )
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not src.exists() or not src.is_dir():
        print(f"ERROR: --src is not a directory: {src}", file=sys.stderr)
        return 2

    target_kbps = args.target_bitrate_kbps
    if target_kbps <= 0:
        print("ERROR: --target-bitrate-kbps must be > 0", file=sys.stderr)
        return 2

    print(f"Scanning: {src}")

    # Collect counts/sizes for ALL extensions (useful inventory),
    # and separately track audio files for bitrate/duration.
    ext_stats: dict[str, ExtStats] = defaultdict(ExtStats)
    audio_files: list[Path] = []

    total_files = 0
    for p in _iter_files(src):
        total_files += 1
        ext = p.suffix.lower() or "<noext>"
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        ext_stats[ext].count += 1
        ext_stats[ext].size_bytes += size

        if p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            audio_files.append(p)

    print(f"Total files: {total_files}")
    print(f"Audio files (supported): {len(audio_files)}")

    # Print extension summary (sorted by size desc).
    print("\n--- EXTENSIONS (all files) ---")
    print(f"{'Ext':<10} {'Count':>10} {'Size':>14}")
    print("-" * 38)
    for ext, st in sorted(ext_stats.items(), key=lambda kv: kv[1].size_bytes, reverse=True):
        print(f"{ext:<10} {st.count:>10} {_human_bytes(st.size_bytes):>14}")

    # Deep scan for audio: duration totals and MP3 bitrate buckets.
    total_audio_duration_sec = 0.0
    mp3_bitrate_buckets: Counter[int] = Counter()
    ffprobe_failures = 0
    probed_audio_bytes = 0
    probed_audio_files = 0
    total_audio_bytes = 0

    for p in audio_files:
        try:
            total_audio_bytes += p.stat().st_size
        except OSError:
            pass

    if audio_files:
        cpu = os.cpu_count() or 4
        workers = args.workers if args.workers and args.workers > 0 else max(1, cpu - 1)
        print("\nProbing audio files with ffprobe (duration/bitrate)...")
        # Thread pool is appropriate here: we spawn many external ffprobe processes,
        # so Python threads work well while waiting on subprocesses.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def probe_one(path: Path) -> tuple[float | None, float | None]:
            info = _run_ffprobe_json(path)
            if not info:
                return None, None
            return _get_format_number(info, "duration"), _get_format_number(info, "bit_rate")

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(probe_one, p): p for p in audio_files}
            for fut in tqdm(as_completed(futures), total=len(futures), unit="file", dynamic_ncols=True):
                p = futures[fut]
                dur, br = fut.result()
                if dur is None or dur <= 0:
                    ffprobe_failures += 1
                    continue

                total_audio_duration_sec += dur
                probed_audio_files += 1
                try:
                    probed_audio_bytes += p.stat().st_size
                except OSError:
                    pass

                if p.suffix.lower() == ".mp3" and br is not None and br > 0:
                    kbps = int(round(br / 1000.0))
                    mp3_bitrate_buckets[kbps] += 1

    if mp3_bitrate_buckets:
        print("\n--- MP3 BITRATE BUCKETS (kbps) ---")
        for kbps in sorted(mp3_bitrate_buckets.keys(), reverse=True):
            print(f"{kbps:>4} kbps : {mp3_bitrate_buckets[kbps]} files")
    else:
        print("\n--- MP3 BITRATE BUCKETS (kbps) ---")
        print("(none found)")

    hours = total_audio_duration_sec / 3600.0
    print("\n--- SUMMARY ---")
    print(f"Audio files (supported): {len(audio_files)}")
    print(f"Current audio size: { _human_bytes(total_audio_bytes) }")
    print(f"Probed (duration OK): {probed_audio_files} files")
    if probed_audio_files:
        print(f"Probed audio size: { _human_bytes(probed_audio_bytes) }")
    print(f"Total audio duration: {hours:.2f} hours")
    if ffprobe_failures:
        print(f"ffprobe failures: {ffprobe_failures} files (excluded from duration/estimate)")

    # Estimate size if everything were target_kbps (bits/sec) => bytes/sec.
    # Use decimal kbps (128,000 bits/sec) as common bitrate unit.
    est_bytes = int(math.floor(total_audio_duration_sec * (target_kbps * 1000) / 8.0))
    print("\n--- ESTIMATE (if all audio were {0} kbps) ---".format(target_kbps))
    print(f"Estimated total size: {_human_bytes(est_bytes)}")

    # Savings: compare estimate to the probed subset size (apples-to-apples)
    if probed_audio_files and probed_audio_bytes > 0:
        saved = probed_audio_bytes - est_bytes
        pct = (saved / probed_audio_bytes) * 100.0
        sign = "-" if saved < 0 else ""
        print(f"Projected savings: {sign}{_human_bytes(abs(saved))} ({pct:.1f}%)")
    elif total_audio_bytes > 0:
        # Fallback if we couldn't probe duration; can't estimate reliably.
        print("Projected savings: (unavailable; could not probe durations)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


