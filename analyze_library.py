from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

# Optional imports for deep ID3 recovery
try:
    import mutagen
    from mutagen.id3 import ID3, ID3NoHeaderError
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False


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

# Common text tag keys we analyze for broken encoding / top values
COMMON_TEXT_TAGS = {
    "title",
    "artist",
    "album",
    "album_artist",
    "albumartist",
    "genre",
    "date",
    "track",
    "disc",
    "composer",
    "comment",
}
# Also include lyrics-* variants (lyrics-eng, lyrics-rus, etc.)
LYRICS_PREFIX = "lyrics"

# Fields for which we print top-20 values
TOP_VALUE_FIELDS = ("title", "artist", "album", "genre")


@dataclass
class ExtStats:
    count: int = 0
    size_bytes: int = 0


@dataclass(frozen=True)
class ProbeResult:
    ffprobe_ok: bool
    duration_sec: float | None
    bitrate_bps: float | None
    tag_keys: frozenset[str]
    embedded_image_count: int
    # tag_values: {normalized_key: raw_value} for common text tags
    tag_values: dict[str, str] = field(default_factory=dict)
    # recovered_values: {normalized_key: recovered_value} from deep ID3 inspection
    recovered_values: dict[str, str] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Mojibake / broken encoding detection + recovery
# ─────────────────────────────────────────────────────────────────────────────

# Cyrillic Unicode block ranges (basic + extended)
_CYRILLIC_RANGES = [
    (0x0400, 0x04FF),  # Cyrillic
    (0x0500, 0x052F),  # Cyrillic Supplement
]


def _is_cyrillic(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CYRILLIC_RANGES)


def _cyrillic_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Cyrillic."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    cyr = sum(1 for c in alpha if _is_cyrillic(c))
    return cyr / len(alpha)


# Common mojibake patterns when UTF-8 bytes are misread as Latin-1/CP1252
_MOJIBAKE_PATTERNS = re.compile(
    r"["
    r"\u00C3\u0192"  # Ã or ƒ often from UTF-8 as Latin-1
    r"\u00C2"        # Â
    r"\u0402-\u040F"  # Cyrillic chars often from CP1251 misread
    r"\u2019\u201C\u201D"  # curly quotes from CP1252
    r"]"
    r"|"
    r"Ð[^a-zA-Z]"  # Ð followed by weird char (UTF-8 Cyrillic prefix)
    r"|"
    r"Ñ[^a-zA-Z]"
)

# Replacement char or lots of control chars
_REPLACEMENT_CHAR = "\uFFFD"


def _looks_broken(text: str) -> bool:
    """
    Heuristic: returns True if text looks like mojibake / bad encoding.
    """
    if not text:
        return False
    # Explicit replacement char
    if _REPLACEMENT_CHAR in text:
        return True
    # Common mojibake byte sequences
    if _MOJIBAKE_PATTERNS.search(text):
        return True
    # High ratio of control chars (excluding normal whitespace)
    ctrl = sum(1 for c in text if unicodedata.category(c) == "Cc" and c not in "\t\n\r")
    if ctrl > len(text) * 0.1:
        return True
    return False


def _try_recover(text: str) -> tuple[str | None, str | None]:
    """
    Attempt to recover a broken string.
    Returns (recovered_text, method_name) or (None, None) if no fix found.

    Strategies:
    1. UTF-8 bytes misread as Latin-1  -> re-encode latin1, decode utf-8
    2. CP1251 bytes misread as Latin-1 -> re-encode latin1, decode cp1251
    """
    if not text or text.strip() == "":
        return None, None

    candidates: list[tuple[str, str]] = []

    # Strategy 1: Latin-1 -> UTF-8
    try:
        raw = text.encode("latin-1", errors="strict")
        decoded = raw.decode("utf-8", errors="strict")
        if decoded != text and not _looks_broken(decoded):
            candidates.append((decoded, "latin1→utf8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Strategy 2: Latin-1 -> CP1251 (common for Russian)
    try:
        raw = text.encode("latin-1", errors="strict")
        decoded = raw.decode("cp1251", errors="strict")
        if decoded != text and not _looks_broken(decoded):
            candidates.append((decoded, "latin1→cp1251"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Strategy 3: CP1252 -> UTF-8 (Windows Western -> UTF-8)
    try:
        raw = text.encode("cp1252", errors="strict")
        decoded = raw.decode("utf-8", errors="strict")
        if decoded != text and not _looks_broken(decoded):
            candidates.append((decoded, "cp1252→utf8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    if not candidates:
        return None, None

    # Prefer candidate with highest Cyrillic ratio (if any), else first
    best = max(candidates, key=lambda c: _cyrillic_ratio(c[0]))
    return best


def _recover_id3_cp1251(path: Path) -> dict[str, str]:
    """
    Uses mutagen to inspect ID3v1/v2 tags directly.
    Tries to interpret Latin-1 frames as CP1251 if they contain high bytes.
    Returns a dict of {normalized_key: recovered_value} for common fields.
    """
    if not HAS_MUTAGEN:
        return {}
    
    recovered = {}
    try:
        tags = ID3(path)
    except (ID3NoHeaderError, Exception):
        return {}

    # Map common ID3 frames to our normalized keys
    frame_map = {
        "TIT2": "title",
        "TPE1": "artist",
        "TALB": "album",
        "TCON": "genre",
        "COMM": "comment",
        # Add others if needed
    }

    for frame_id, key in frame_map.items():
        frames = tags.getall(frame_id)
        for frame in frames:
            # We are looking for frames where encoding=0 (Latin-1) but bytes are likely CP1251
            if getattr(frame, "encoding", -1) == 0:  # 0 is Latin-1 in ID3v2
                # mutagen automatically decodes Latin-1 to unicode string
                # So we take that string, re-encode to latin-1 to get bytes, then try cp1251
                val_list = getattr(frame, "text", [])
                if not val_list:
                    continue
                
                # Handling COMM frames which are a bit different
                if frame_id == "COMM":
                    text_str = frame.text[0] if frame.text else ""
                else:
                    text_str = val_list[0]

                if not text_str:
                    continue

                # Check if it looks like garbage or has replacement chars
                if _looks_broken(text_str) or _REPLACEMENT_CHAR in text_str:
                     # Try to recover by re-interpreting bytes
                    rec, method = _try_recover(text_str)
                    if rec:
                        recovered[key] = rec
    return recovered



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
        "-show_format",
        "-show_streams",
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


def _is_common_text_tag(key: str) -> bool:
    """Check if key is a common text tag we want to analyze."""
    kl = key.lower()
    if kl in COMMON_TEXT_TAGS:
        return True
    if kl.startswith(LYRICS_PREFIX):
        return True
    return False


def _extract_tag_keys_and_values(info: dict[str, Any] | None) -> tuple[frozenset[str], dict[str, str]]:
    """
    Returns:
      - A normalized (lowercased) set of metadata tag keys present in the file.
      - A dict of {normalized_key: raw_value} for common text tags only.
    Includes both container-level format tags and stream-level tags (if present).
    """
    if not info:
        return frozenset(), {}

    keys: set[str] = set()
    values: dict[str, str] = {}

    def process_tags(tags: dict[str, Any]) -> None:
        for k, v in tags.items():
            ks = str(k).strip().lower()
            if not ks:
                continue
            keys.add(ks)
            # Collect value for common text tags (first occurrence wins)
            if _is_common_text_tag(ks) and ks not in values:
                vs = str(v).strip() if v is not None else ""
                if vs:
                    values[ks] = vs

    fmt = info.get("format") or {}
    fmt_tags = fmt.get("tags") or {}
    if isinstance(fmt_tags, dict):
        process_tags(fmt_tags)

    streams = info.get("streams") or []
    if isinstance(streams, list):
        for st in streams:
            if not isinstance(st, dict):
                continue
            st_tags = st.get("tags") or {}
            if isinstance(st_tags, dict):
                process_tags(st_tags)

    return frozenset(keys), values


def _count_embedded_images(info: dict[str, Any] | None) -> int:
    """
    Counts embedded image streams using ffprobe's `streams[].disposition.attached_pic`.
    This typically detects cover art for MP3/M4A/FLAC, etc., when represented as an attached picture stream.
    """
    if not info:
        return 0

    streams = info.get("streams") or []
    if not isinstance(streams, list):
        return 0

    n = 0
    for st in streams:
        if not isinstance(st, dict):
            continue
        disp = st.get("disposition") or {}
        if not isinstance(disp, dict):
            continue
        attached = disp.get("attached_pic")
        if attached in (1, "1", True):
            n += 1
    return n


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
    ffprobe_ok_files = 0
    total_audio_bytes = 0
    tag_key_counts: Counter[str] = Counter()
    embedded_images_by_ext: Counter[str] = Counter()
    files_with_images_by_ext: Counter[str] = Counter()
    total_embedded_images = 0
    files_with_any_images = 0

    # Top value counters for title, artist, album, genre
    top_values: dict[str, Counter[str]] = {f: Counter() for f in TOP_VALUE_FIELDS}

    # Broken encoding tracking
    # broken_by_field[field] = list of (path, raw_value, recovered_value_or_None, method_or_None)
    broken_by_field: dict[str, list[tuple[Path, str, str | None, str | None]]] = defaultdict(list)

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

        def probe_one(path: Path) -> ProbeResult:
            info = _run_ffprobe_json(path)
            if not info:
                return ProbeResult(
                    ffprobe_ok=False,
                    duration_sec=None,
                    bitrate_bps=None,
                    tag_keys=frozenset(),
                    embedded_image_count=0,
                    tag_values={},
                    recovered_values={},
                )

            dur = _get_format_number(info, "duration")
            br = _get_format_number(info, "bit_rate")
            tag_keys, tag_values = _extract_tag_keys_and_values(info)

            recovered_deep = {}
            # If MP3 and we have mutagen, try to recover broken tags directly from ID3
            if HAS_MUTAGEN and path.suffix.lower() == ".mp3":
                # Check if we have any broken tags worth recovering
                needs_recovery = any(_looks_broken(v) for v in tag_values.values())
                if needs_recovery:
                    recovered_deep = _recover_id3_cp1251(path)

            embedded_image_count = _count_embedded_images(info)
            return ProbeResult(
                ffprobe_ok=True,
                duration_sec=dur,
                bitrate_bps=br,
                tag_keys=tag_keys,
                embedded_image_count=embedded_image_count,
                tag_values=tag_values,
                recovered_values=recovered_deep,
            )

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(probe_one, p): p for p in audio_files}
            for fut in tqdm(as_completed(futures), total=len(futures), unit="file", dynamic_ncols=True):
                p = futures[fut]
                res = fut.result()
                if not res.ffprobe_ok:
                    ffprobe_failures += 1
                    continue

                ffprobe_ok_files += 1
                # Tag keys (count once per file per key)
                for k in res.tag_keys:
                    tag_key_counts[k] += 1

                # Top values for title, artist, album, genre
                # Prefer recovered values for stats if available
                for fld in TOP_VALUE_FIELDS:
                    val = res.tag_values.get(fld)
                    if not val:
                        continue

                    # 1. Try deep recovery (ID3 byte inspection)
                    rec = res.recovered_values.get(fld)
                    
                    # 2. If no deep recovery, try heuristic recovery (for mojibake)
                    if not rec and _looks_broken(val):
                        rec_heuristic, _ = _try_recover(val)
                        if rec_heuristic:
                            rec = rec_heuristic

                    # Use recovered if available, else original
                    final_val = rec if rec else val
                    top_values[fld][final_val] += 1

                # Broken encoding detection for common text tags
                for fld, val in res.tag_values.items():
                    if _looks_broken(val):
                        # 1. Try deep recovery (mutagen ID3 inspection)
                        deep_rec = res.recovered_values.get(fld)
                        if deep_rec:
                             broken_by_field[fld].append((p, val, deep_rec, "ID3-CP1251"))
                        else:
                             # 2. Try heuristic recovery
                             recovered, method = _try_recover(val)
                             broken_by_field[fld].append((p, val, recovered, method))

                # Embedded images (cover art) per extension
                ext = p.suffix.lower() or "<noext>"
                if res.embedded_image_count > 0:
                    files_with_any_images += 1
                    files_with_images_by_ext[ext] += 1
                    embedded_images_by_ext[ext] += res.embedded_image_count
                    total_embedded_images += res.embedded_image_count

                if res.duration_sec is not None and res.duration_sec > 0:
                    total_audio_duration_sec += res.duration_sec
                    probed_audio_files += 1
                    try:
                        probed_audio_bytes += p.stat().st_size
                    except OSError:
                        pass

                if p.suffix.lower() == ".mp3" and res.bitrate_bps is not None and res.bitrate_bps > 0:
                    kbps = int(round(res.bitrate_bps / 1000.0))
                    mp3_bitrate_buckets[kbps] += 1

    if mp3_bitrate_buckets:
        print("\n--- MP3 BITRATE BUCKETS (kbps) ---")
        for kbps in sorted(mp3_bitrate_buckets.keys(), reverse=True):
            print(f"{kbps:>4} kbps : {mp3_bitrate_buckets[kbps]} files")
    else:
        print("\n--- MP3 BITRATE BUCKETS (kbps) ---")
        print("(none found)")

    if tag_key_counts and ffprobe_ok_files:
        print("\n--- TAG KEYS (files containing key) ---")
        print(f"{'Key':<26} {'Files':>10} {'Pct':>7}")
        print("-" * 45)
        for k, c in sorted(tag_key_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            pct = (c / ffprobe_ok_files) * 100.0
            print(f"{k:<26} {c:>10} {pct:>6.1f}%")
    else:
        print("\n--- TAG KEYS (files containing key) ---")
        print("(none found)")

    # Broken encoding summary
    total_broken = sum(len(v) for v in broken_by_field.values())
    total_recoverable = sum(
        sum(1 for _, _, rec, _ in v if rec is not None)
        for v in broken_by_field.values()
    )
    print("\n--- BROKEN ENCODING ANALYSIS ---")
    print(f"Total files with broken text tags: {total_broken}")
    print(f"Total likely recoverable: {total_recoverable}")
    if broken_by_field:
        print(f"\n{'Field':<20} {'Broken':>8} {'Recoverable':>12}")
        print("-" * 42)
        for fld in sorted(broken_by_field.keys(), key=lambda f: -len(broken_by_field[f])):
            items = broken_by_field[fld]
            rec_count = sum(1 for _, _, r, _ in items if r is not None)
            print(f"{fld:<20} {len(items):>8} {rec_count:>12}")

        # Sample broken values (top 20 overall, grouped by field)
        print("\n--- SAMPLE BROKEN VALUES (up to 20) ---")
        samples_shown = 0
        for fld in sorted(broken_by_field.keys(), key=lambda f: -len(broken_by_field[f])):
            if samples_shown >= 20:
                break
            items = broken_by_field[fld]
            for path, raw, recovered, method in items[:max(1, 20 - samples_shown)]:
                samples_shown += 1
                rel_path = path.name  # just filename for brevity
                raw_short = raw if len(raw) <= 40 else raw[:37] + "..."
                if recovered:
                    rec_short = recovered if len(recovered) <= 40 else recovered[:37] + "..."
                    print(f"  [{fld}] {rel_path}")
                    print(f"    Raw:       {raw_short}")
                    print(f"    Recovered: {rec_short} ({method})")
                else:
                    print(f"  [{fld}] {rel_path}")
                    print(f"    Raw:       {raw_short}")
                    print(f"    Recovered: (no fix found)")
                if samples_shown >= 20:
                    break
    else:
        print("(no broken encoding detected)")

    # Top 20 values for title, artist, album, genre
    for fld in TOP_VALUE_FIELDS:
        counter = top_values[fld]
        print(f"\n--- TOP 20 VALUES: {fld.upper()} ---")
        if counter:
            print(f"{'Value':<50} {'Count':>8}")
            print("-" * 60)
            for val, cnt in counter.most_common(20):
                # Truncate long values for display
                display_val = val if len(val) <= 48 else val[:45] + "..."
                print(f"{display_val:<50} {cnt:>8}")
        else:
            print("(none found)")

    print("\n--- EMBEDDED IMAGES (attached_pic streams) ---")
    if ffprobe_ok_files:
        print(f"Files with embedded images: {files_with_any_images}/{ffprobe_ok_files}")
    else:
        print("Files with embedded images: (unavailable; no ffprobe results)")
    print(f"Total embedded images: {total_embedded_images}")
    if embedded_images_by_ext or files_with_images_by_ext:
        print(f"{'Ext':<10} {'FilesWithArt':>14} {'Images':>10}")
        print("-" * 38)
        all_exts = set(embedded_images_by_ext.keys()) | set(files_with_images_by_ext.keys())
        for ext in sorted(all_exts, key=lambda e: (-files_with_images_by_ext.get(e, 0), e)):
            print(f"{ext:<10} {files_with_images_by_ext.get(ext, 0):>14} {embedded_images_by_ext.get(ext, 0):>10}")
    else:
        print("(none found)")

    hours = total_audio_duration_sec / 3600.0
    print("\n--- SUMMARY ---")
    print(f"Audio files (supported): {len(audio_files)}")
    print(f"Current audio size: { _human_bytes(total_audio_bytes) }")
    if audio_files:
        print(f"ffprobe OK (metadata): {ffprobe_ok_files} files")
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


