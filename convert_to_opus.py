from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

from mutagen import File as MutagenFile
from mutagen.flac import Picture
from mutagen.id3 import APIC, ID3
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


def run_ffmpeg_convert(src: Path, dst: Path, bitrate: str) -> Tuple[bool, str]:
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
        "-vn",
        str(dst),
    ]

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


def process_job(job: Job, bitrate: str) -> JobResult:
    job.dst.parent.mkdir(parents=True, exist_ok=True)

    ok, err = run_ffmpeg_convert(job.src, job.dst, bitrate)
    if not ok:
        return JobResult(src=job.src, dst=job.dst, ok=False, error=err)

    # Cover art: try mutagen, fallback to ffmpeg extraction.
    cover = extract_cover_with_mutagen(job.src) or extract_cover_with_ffmpeg(job.src)
    if cover:
        mime, data = cover
        try:
            embed_cover_into_opus(job.dst, mime, data)
            return JobResult(src=job.src, dst=job.dst, ok=True, cover_found=True, cover_embedded=True)
        except Exception:
            # Don't fail the whole conversion if cover embedding fails.
            return JobResult(
                src=job.src,
                dst=job.dst,
                ok=True,
                cover_found=True,
                cover_embedded=False,
                cover_error="embed_failed",
            )

    return JobResult(src=job.src, dst=job.dst, ok=True, cover_found=False, cover_embedded=False)


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
            collisions.setdefault(out_path, [out_map[out_path]]).append(p)
        else:
            out_map[out_path] = p
            jobs.append(Job(src=p, dst=out_path))

    if collisions:
        print("\nERROR: Output name collisions detected (same output .opus path from multiple inputs).", file=sys.stderr)
        for out_path, inputs in list(collisions.items())[:20]:
            print(f"  {out_path} <-", file=sys.stderr)
            for inp in inputs:
                print(f"    - {inp}", file=sys.stderr)
        if len(collisions) > 20:
            print(f"  ... and {len(collisions) - 20} more collisions", file=sys.stderr)
        print("Fix by renaming duplicates in source (e.g., same song name in same folder with different extensions).", file=sys.stderr)
        return 2

    # Convert in parallel
    errors: list[JobResult] = []
    ok_count = 0
    cover_found_count = 0
    cover_missing_count = 0
    cover_embedded_count = 0
    cover_embed_failed_count = 0

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


