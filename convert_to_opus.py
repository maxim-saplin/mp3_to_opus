from __future__ import annotations

import argparse
import base64
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

from mutagen import File as MutagenFile
from mutagen.flac import Picture
from mutagen.id3 import APIC, ID3, ID3NoHeaderError
from mutagen.mp4 import MP4, MP4Cover
from mutagen.oggopus import OggOpus
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

IGNORED_SIDE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".txt",
    ".nfo",
    ".m3u",
    ".m3u8",
    ".log",
    ".cue",
    ".ini",
    ".db",
}


# ─────────────────────────────────────────────────────────────────────────────
# Text Recovery Utils (adapted from analyze_library.py)
# ─────────────────────────────────────────────────────────────────────────────

_CYRILLIC_RANGES = [(0x0400, 0x04FF), (0x0500, 0x052F)]
_MOJIBAKE_PATTERNS = re.compile(
    r"[\u00C3\u0192\u00C2\u0402-\u040F\u2019\u201C\u201D]|Ð[^a-zA-Z]|Ñ[^a-zA-Z]"
)
_REPLACEMENT_CHAR = "\uFFFD"


def _is_cyrillic(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CYRILLIC_RANGES)


def _cyrillic_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    cyr = sum(1 for c in alpha if _is_cyrillic(c))
    return cyr / len(alpha)


def _looks_broken(text: str) -> bool:
    if not text:
        return False
    if _REPLACEMENT_CHAR in text:
        return True
    if _MOJIBAKE_PATTERNS.search(text):
        return True
    ctrl = sum(1 for c in text if unicodedata.category(c) == "Cc" and c not in "\t\n\r")
    if ctrl > len(text) * 0.1:
        return True
    return False


def _try_recover(text: str) -> str | None:
    """Returns recovered text or None."""
    if not text or text.strip() == "":
        return None
    
    candidates = []
    # 1. Latin-1 -> UTF-8
    try:
        dec = text.encode("latin-1").decode("utf-8")
        if dec != text and not _looks_broken(dec):
            candidates.append(dec)
    except Exception:
        pass
    # 2. Latin-1 -> CP1251
    try:
        dec = text.encode("latin-1").decode("cp1251")
        if dec != text and not _looks_broken(dec):
            candidates.append(dec)
    except Exception:
        pass
    # 3. CP1252 -> UTF-8
    try:
        dec = text.encode("cp1252").decode("utf-8")
        if dec != text and not _looks_broken(dec):
            candidates.append(dec)
    except Exception:
        pass

    if not candidates:
        return None
    return max(candidates, key=_cyrillic_ratio)


def _recover_id3_cp1251(path: Path) -> dict[str, str]:
    """Inspect ID3 tags for Latin-1 encoded CP1251 text."""
    try:
        tags = ID3(path)
    except (ID3NoHeaderError, Exception):
        return {}

    recovered = {}
    # Map ID3 frames to ffmpeg metadata keys
    frame_map = {
        "TIT2": "title",
        "TPE1": "artist",
        "TALB": "album",
        "TCON": "genre",
    }

    for frame_id, key in frame_map.items():
        frames = tags.getall(frame_id)
        for frame in frames:
            if getattr(frame, "encoding", -1) == 0:  # Latin-1
                text_list = getattr(frame, "text", [])
                text = text_list[0] if text_list else ""
                if text and (_looks_broken(text) or _REPLACEMENT_CHAR in text):
                    rec = _try_recover(text)
                    if rec:
                        recovered[key] = rec
    return recovered


def get_smart_metadata(src: Path) -> Tuple[dict[str, str], bool, bool]:
    """
    Returns (meta_overrides, recovered_flag, filled_flag).
    """
    meta = {}
    recovered = False
    filled = False
    
    # Read raw tags first
    try:
        f = MutagenFile(src)
        if f:
            pass
    except Exception:
        pass

    # For MP3s, we might have deeply recovered tags
    if src.suffix.lower() == ".mp3":
        recovered_id3 = _recover_id3_cp1251(src)
        if recovered_id3:
            meta.update(recovered_id3)
            recovered = True

    # We need "current" title/artist to decide if we should overwrite with filename
    # Simple extraction:
    cur_title = meta.get("title")
    cur_artist = meta.get("artist")
    
    # If not recovered yet, try reading standard tags
    if not cur_title or not cur_artist:
        try:
            f = MutagenFile(src)
            if f and f.tags:
                # easy access helper
                def get_tag(k_list):
                    for k in k_list:
                        if k in f.tags:
                            val = f.tags[k]
                            if isinstance(val, list) and val:
                                v = val[0]
                                if hasattr(v, "text"):
                                    return v.text[0]
                                return str(v)
                    return None

                if not cur_title:
                    t = get_tag(["title", "TITLE", "TIT2", "©nam"])
                    if t: cur_title = t
                if not cur_artist:
                    a = get_tag(["artist", "ARTIST", "TPE1", "©ART"])
                    if a: cur_artist = a
        except Exception:
            pass

    # Heuristic fix for current tags if they exist but are broken
    if cur_title and _looks_broken(cur_title):
        rec = _try_recover(cur_title)
        if rec:
            cur_title = rec
            recovered = True
    if cur_artist and _looks_broken(cur_artist):
        rec = _try_recover(cur_artist)
        if rec:
            cur_artist = rec
            recovered = True

    # Check for missing/generic
    # Generic patterns: "Track 1", "no title", empty
    generic_re = re.compile(r"^(track\s*\d+|no\s*title|unknown|artist|title)$", re.IGNORECASE)

    is_title_bad = not cur_title or not cur_title.strip() or generic_re.match(cur_title.strip())
    is_artist_bad = not cur_artist or not cur_artist.strip() or generic_re.match(cur_artist.strip())

    # If bad, try filename parse: "Artist - Title.ext"
    if is_title_bad or is_artist_bad:
        # e.g. "Dj Smash - Между небом и землей.mp3"
        stem = src.stem  # "Dj Smash - Между небом и землей"
        # Match "Artist - Title"
        # We assume " - " separator.
        parts = stem.split(" - ", 1)
        if len(parts) == 2:
            fname_artist, fname_title = parts[0].strip(), parts[1].strip()
            if is_artist_bad and fname_artist:
                cur_artist = fname_artist
                filled = True
            if is_title_bad and fname_title:
                cur_title = fname_title
                filled = True
    
    # Update meta dict with final values if they differ from what might be in the file
    if cur_title:
        meta["title"] = cur_title
    if cur_artist:
        meta["artist"] = cur_artist

    return meta, recovered, filled


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    n = float(max(0, num_bytes))
    for u in units:
        if n < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(n)} {u}"
            return f"{n:.2f} {u}"
        n /= 1024.0
    return f"{num_bytes} B"


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def ensure_empty_dir(dst: Path) -> None:
    if dst.exists():
        # Destination must be empty
        try:
            next(dst.iterdir())
            raise RuntimeError(f"Destination is not empty: {dst}")
        except StopIteration:
            return
    dst.mkdir(parents=True, exist_ok=True)


def is_audio_file(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_AUDIO_EXTS and p.suffix.lower() != ".opus"


def classify_non_audio(p: Path) -> str:
    """
    Returns: 'ignored' for common sidecar files, 'warn' otherwise.
    """
    if p.suffix.lower() in IGNORED_SIDE_EXTS:
        return "ignored"
    return "warn"


def run_ffmpeg_convert(src: Path, dst: Path, bitrate: str, metadata: dict[str, str] | None = None) -> Tuple[bool, str]:
    """
    Returns (ok, message). If failed, message contains stderr (best-effort).
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-n",  # do NOT overwrite
        "-i",
        str(src),
        "-map",
        "0:a:0",
        "-c:a",
        "libopus",
        "-b:a",
        bitrate,
        "-vbr",
        "on",
        "-compression_level",
        "10",
        "-application",
        "audio",
        "-map_metadata",
        "0",
    ]

    # Add metadata overrides
    if metadata:
        for k, v in metadata.items():
            if v:
                cmd.extend(["-metadata", f"{k}={v}"])

    cmd.extend([
        "-vn",
        str(dst),
    ])

    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError:
        return False, "ffmpeg not found on PATH"

    if proc.returncode == 0:
        return True, ""
    # ffmpeg prints errors to stderr
    err = proc.stderr.decode("utf-8", errors="replace")
    return False, err.strip()


def extract_cover_with_mutagen(src: Path) -> Optional[Tuple[str, bytes]]:
    """
    Best-effort cover extraction via mutagen.
    Returns (mime, data) or None.
    """
    try:
        # Special-case MP3 with ID3 APIC for reliability
        if src.suffix.lower() == ".mp3":
            id3 = ID3(src)
            apics = [f for f in id3.values() if isinstance(f, APIC)]
            if apics:
                apic = apics[0]
                mime = apic.mime or "image/jpeg"
                return mime, bytes(apic.data)

        if src.suffix.lower() in {".m4a", ".mp4"}:
            mp4 = MP4(src)
            covr = mp4.tags.get("covr") if mp4.tags else None
            if covr:
                cover = covr[0]
                if isinstance(cover, MP4Cover):
                    if cover.imageformat == MP4Cover.FORMAT_PNG:
                        return "image/png", bytes(cover)
                    return "image/jpeg", bytes(cover)
                return "image/jpeg", bytes(cover)

        # Generic: FLAC, OGG, etc.
        mf = MutagenFile(src)
        if mf is None:
            return None

        # FLAC
        pics = getattr(mf, "pictures", None)
        if pics:
            pic0 = pics[0]
            mime = getattr(pic0, "mime", None) or "image/jpeg"
            data = getattr(pic0, "data", None)
            if data:
                return mime, bytes(data)

        # Some formats may expose 'APIC:'-like keys, but parsing varies; stop here.
        return None
    except Exception:
        return None


def extract_cover_with_ffmpeg(src: Path) -> Optional[Tuple[str, bytes]]:
    """
    Fallback cover extraction using ffmpeg. Always produces JPEG bytes if a cover exists.
    Returns (mime, data) or None.
    """
    with tempfile.TemporaryDirectory() as td:
        out_jpg = Path(td) / "cover.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(src),
            "-an",
            "-map",
            "0:v:0?",
            "-frames:v",
            "1",
            str(out_jpg),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, check=False)
        except FileNotFoundError:
            return None
        if proc.returncode != 0 or not out_jpg.exists():
            return None
        try:
            data = out_jpg.read_bytes()
        except OSError:
            return None
        if not data:
            return None
        return "image/jpeg", data


def embed_cover_into_opus(dst_opus: Path, mime: str, data: bytes) -> None:
    """
    Embed cover art into an .opus (OggOpus) file using METADATA_BLOCK_PICTURE.
    """
    opus = OggOpus(dst_opus)
    if opus.tags is None:
        opus.add_tags()

    pic = Picture()
    pic.type = 3  # front cover
    pic.desc = "Cover (front)"
    pic.mime = mime or "image/jpeg"
    pic.data = data
    b64 = base64.b64encode(pic.write()).decode("ascii")

    # Replace any existing cover(s)
    # Note: mutagen's Vorbis comment dict behaves like a mapping but its pop()
    # does not accept a default value on some versions.
    try:
        del opus.tags["METADATA_BLOCK_PICTURE"]
    except KeyError:
        pass
    opus.tags["METADATA_BLOCK_PICTURE"] = [b64]
    opus.save()


@dataclass
class Job:
    src: Path
    dst: Path


@dataclass
class JobResult:
    src: Path
    dst: Path
    ok: bool
    error: str = ""
    cover_found: bool = False
    cover_embedded: bool = False
    cover_error: str = ""
    tags_recovered: bool = False
    tags_filled: bool = False


def process_job(job: Job, bitrate: str) -> JobResult:
    job.dst.parent.mkdir(parents=True, exist_ok=True)

    # Determine smart metadata (fix broken tags, fill from filename)
    meta, recovered, filled = get_smart_metadata(job.src)

    ok, err = run_ffmpeg_convert(job.src, job.dst, bitrate, metadata=meta)
    if not ok:
        return JobResult(src=job.src, dst=job.dst, ok=False, error=err)

    # Cover art: try mutagen, fallback to ffmpeg extraction.
    cover = extract_cover_with_mutagen(job.src) or extract_cover_with_ffmpeg(job.src)
    if cover:
        mime, data = cover
        try:
            embed_cover_into_opus(job.dst, mime, data)
            return JobResult(
                src=job.src,
                dst=job.dst,
                ok=True,
                cover_found=True,
                cover_embedded=True,
                tags_recovered=recovered,
                tags_filled=filled,
            )
        except Exception:
            # Don't fail the whole conversion if cover embedding fails.
            return JobResult(
                src=job.src,
                dst=job.dst,
                ok=True,
                cover_found=True,
                cover_embedded=False,
                cover_error="embed_failed",
                tags_recovered=recovered,
                tags_filled=filled,
            )

    return JobResult(
        src=job.src,
        dst=job.dst,
        ok=True,
        cover_found=False,
        cover_embedded=False,
        tags_recovered=recovered,
        tags_filled=filled,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert an audio library to Opus while preserving folder structure and metadata.")
    parser.add_argument("--src", required=True, help="Source folder (recursive).")
    parser.add_argument("--dst", required=True, help="Destination folder (must be empty or not exist).")
    parser.add_argument("--bitrate", default="128k", help="Opus target bitrate (default: 128k).")
    parser.add_argument("--workers", type=int, default=0, help="Number of parallel workers (default: cpu_count-1).")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    bitrate = args.bitrate

    if not src.exists() or not src.is_dir():
        print(f"ERROR: --src is not a directory: {src}", file=sys.stderr)
        return 2

    try:
        ensure_empty_dir(dst)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Worker default
    cpu = os.cpu_count() or 4
    workers = args.workers if args.workers and args.workers > 0 else max(1, cpu - 1)

    # Scan & prepare jobs
    all_files = list(iter_files(src))
    audio_files: list[Path] = []
    warn_non_audio = 0

    for p in all_files:
        if is_audio_file(p):
            audio_files.append(p)
        else:
            if classify_non_audio(p) == "warn":
                warn_non_audio += 1
                print(f"WARNING: skipping non-audio file: {p}")

    if not audio_files:
        print("No supported audio files found. Nothing to do.")
        return 0

    # Pre-flight stats
    total_src_bytes = 0
    for p in audio_files:
        try:
            total_src_bytes += p.stat().st_size
        except OSError:
            pass

    print("\n--- PRE-FLIGHT ---")
    print(f"Source: {src}")
    print(f"Destination: {dst}")
    print(f"Audio files: {len(audio_files)}")
    print(f"Total audio size: {human_bytes(total_src_bytes)}")
    print(f"Workers: {workers}")

    # Build output mapping and detect collisions
    out_map: dict[Path, Path] = {}
    collisions: dict[Path, list[Path]] = {}
    jobs: list[Job] = []

    for p in audio_files:
        rel = p.relative_to(src)
        out_rel = rel.with_suffix(".opus")
        out_path = dst / out_rel
        if out_path in out_map:
            # Collision detected: record it but don't add job (first wins)
            collisions.setdefault(out_path, [out_map[out_path]]).append(p)
        else:
            out_map[out_path] = p
            jobs.append(Job(src=p, dst=out_path))

    if collisions:
        print("\nWARNING: Output name collisions detected. Using first file, skipping others.", file=sys.stderr)
        for out_path, inputs in list(collisions.items())[:20]:
            winner = inputs[0]
            losers = inputs[1:]
            print(f"  {out_path}", file=sys.stderr)
            print(f"    + Using: {winner}", file=sys.stderr)
            for l in losers:
                print(f"    - Skip:  {l}", file=sys.stderr)
        if len(collisions) > 20:
            print(f"  ... and {len(collisions) - 20} more collisions", file=sys.stderr)

    # Convert in parallel
    errors: list[JobResult] = []
    ok_count = 0
    cover_found_count = 0
    cover_missing_count = 0
    cover_embedded_count = 0
    cover_embed_failed_count = 0
    tags_recovered_count = 0
    tags_filled_count = 0

    print("\nConverting...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_job, j, bitrate): j for j in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="file", dynamic_ncols=True):
            res = fut.result()
            if res.ok:
                ok_count += 1
                if res.cover_found:
                    cover_found_count += 1
                    if res.cover_embedded:
                        cover_embedded_count += 1
                    else:
                        cover_embed_failed_count += 1
                else:
                    cover_missing_count += 1
                
                if res.tags_recovered:
                    tags_recovered_count += 1
                if res.tags_filled:
                    tags_filled_count += 1
            else:
                errors.append(res)

    # Post-flight stats
    opus_files = list(dst.rglob("*.opus"))
    total_dst_bytes = 0
    for p in opus_files:
        try:
            total_dst_bytes += p.stat().st_size
        except OSError:
            pass

    print("\n--- POST-FLIGHT ---")
    print(f"Converted OK: {ok_count}/{len(jobs)}")
    print(f"Cover found: {cover_found_count}/{len(jobs)}")
    print(f"Cover missing: {cover_missing_count}/{len(jobs)}")
    print(f"Cover embedded: {cover_embedded_count}/{len(jobs)} (best-effort)")
    if cover_embed_failed_count:
        print(f"Cover embed failed: {cover_embed_failed_count}/{len(jobs)} (best-effort)")
    
    print(f"Tags recovered: {tags_recovered_count}/{len(jobs)} (encoding fix)")
    print(f"Tags filled: {tags_filled_count}/{len(jobs)} (from filename)")

    if errors:
        print(f"Errors: {len(errors)}")
        print("First errors:")
        for e in errors[:10]:
            msg = e.error.splitlines()[-1] if e.error else "unknown error"
            print(f"  - {e.src.name}: {msg}")
    else:
        print("Errors: 0")
    print("")
    print(f"Output .opus files: {len(opus_files)}")
    print(f"Output total size: {human_bytes(total_dst_bytes)}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())


