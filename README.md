# Audio Library → Opus (Python + uv)

This repo contains two Python scripts to **analyze** and **convert** a music library to **Opus 128 kbps (VBR)** on Windows, while:
- Preserving **folder structure**
- Preserving **text metadata** (Artist/Album/Title/etc.) via FFmpeg
- Preserving **cover art** (best-effort) via `mutagen`
- Showing **progress % + ETA**
- Printing **pre-flight** (src count/size) and **post-flight** (dst count/size) summaries

## Scripts

### `analyze_library.py`
- Scans your library and prints:\n  - **All file extensions** (count + total size)\n  - **MP3 bitrate buckets** (how many 320 kbps / 192 kbps / etc.)\n  - **Estimated total size** if all audio were encoded at 128 kbps

### `convert_to_opus.py`
- Converts supported audio formats to `.opus` under a destination folder, mirroring structure.\n- Runs multiple conversions in parallel (default: **CPU count - 1**).\n- **Safety**: destination must be **empty or not exist** (script exits if not empty).\n- Warns on unknown non-audio files (sidecars like `.jpg`, `.m3u`, `.cue` are ignored quietly).

## Prerequisites

### 1) FFmpeg on PATH
You need `ffmpeg` and `ffprobe` available on PATH.

Verify:
```powershell
ffmpeg -version
ffprobe -version
```

### 2) Python + uv
You said you have `uv` installed. This repo uses a `.venv` created by `uv`.

## Setup
From the repo folder:

```powershell
uv venv
uv sync
```

## Usage

### Analyze (recommended first)
```powershell
uv run python analyze_library.py --src "G:\Music"
```

### Convert
Pick an **empty** destination folder (or a folder that does not exist yet):
```powershell
uv run python convert_to_opus.py --src "G:\Music" --dst "G:\Converted"
```

Optional tuning:
```powershell
uv run python convert_to_opus.py --src "G:\Music" --dst "G:\Converted" --workers 8
uv run python convert_to_opus.py --src "G:\Music" --dst "G:\Converted" --bitrate 160k
```

## Notes
- **Unicode filenames (Cyrillic, etc.)**: Python handles these natively; FFmpeg is invoked with Unicode-safe argument lists.\n- **Opus & 44.1 kHz**: Opus operates internally at 48 kHz, but 44.1 kHz sources are fine; the encoder/player handles resampling.\n- **Cover art**: embedded best-effort using `METADATA_BLOCK_PICTURE` in the output `.opus`. Some unusual source formats may not expose art cleanly.\n- **Legacy PowerShell scripts**: `opus_converter.ps1` / `analyze_library.ps1` are still in the repo but the Python tools are the recommended path.
