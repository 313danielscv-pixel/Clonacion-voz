"""Shared local audio processing used by the notebook and Streamlit app."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 44_100
VOICE_SAMPLE_RATE = 24_000
MAX_REFERENCE_SECONDS = 25
MAX_SOURCE_SECONDS = 300


class AudioPipelineError(RuntimeError):
    """An audio tool or model could not complete a pipeline step."""


def _run(command: list[str], operation: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioPipelineError(f"{command[0]} is not installed or is missing from PATH.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise AudioPipelineError(f"{operation} failed.\n{detail[-2000:]}") from exc


def probe_duration(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        "Reading media duration",
    )
    duration = float(result.stdout.strip())
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"Could not determine a valid duration for {path.name}.")
    return duration


def prepare_source_audio(
    source: Path,
    destination: Path,
    *,
    limit_seconds: int | None = None,
) -> float:
    duration = probe_duration(source)
    if duration > MAX_SOURCE_SECONDS:
        raise ValueError(
            f"The source is {duration / 60:.1f} minutes long. "
            f"The local app limit is {MAX_SOURCE_SECONDS // 60} minutes."
        )
    if limit_seconds is not None and not 1 <= limit_seconds <= MAX_SOURCE_SECONDS:
        raise ValueError(f"Audio limit must be between 1 and {MAX_SOURCE_SECONDS} seconds.")
    processed_duration = min(duration, limit_seconds) if limit_seconds is not None else duration
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
    ]
    if limit_seconds is not None:
        command.extend(["-t", str(limit_seconds)])
    command.extend(
        [
            "-ac",
            "2",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )
    _run(command, "Preparing the song or mix")
    return processed_duration


def prepare_voice_reference(
    source: Path,
    destination: Path,
    *,
    clean: bool = True,
    max_seconds: int = MAX_REFERENCE_SECONDS,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Voice recording not found: {source}")
    if max_seconds < 1 or max_seconds > MAX_REFERENCE_SECONDS:
        raise ValueError(f"Voice reference duration must be 1-{MAX_REFERENCE_SECONDS} seconds.")

    filters = ["highpass=f=75", "lowpass=f=11000"]
    if clean:
        filters.append("afftdn=nr=8:nf=-45")
    filters.append("loudnorm=I=-24:TP=-2:LRA=7")

    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-t",
            str(max_seconds),
            "-ac",
            "1",
            "-ar",
            str(VOICE_SAMPLE_RATE),
            "-af",
            ",".join(filters),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        "Cleaning and normalizing the voice reference",
    )

    samples, sample_rate = sf.read(destination, dtype="float32")
    if sample_rate != VOICE_SAMPLE_RATE or samples.size == 0 or not np.isfinite(samples).all():
        raise AudioPipelineError("The prepared voice reference is empty or invalid.")
    if float(np.sqrt(np.mean(np.square(samples)))) < 1e-5:
        raise ValueError("The voice reference is silent. Try a clearer recording.")


def separate_vocal_stems(source: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems",
            "vocals",
            "-n",
            "htdemucs",
            "-o",
            str(output_dir),
            str(source),
        ],
        "Separating vocals from the accompaniment",
    )
    stem_dir = output_dir / "htdemucs" / source.stem
    vocals = stem_dir / "vocals.wav"
    accompaniment = stem_dir / "no_vocals.wav"
    if not vocals.is_file() or not accompaniment.is_file():
        raise AudioPipelineError(f"Demucs did not create both expected tracks in {stem_dir}.")
    return vocals, accompaniment


def load_seedvc_model(repo_path: Path, device: torch.device):
    wrapper_path = repo_path / "seed_vc_wrapper.py"
    if not wrapper_path.is_file():
        raise FileNotFoundError(
            f"Seed-VC was not found at {repo_path}. Run the notebook setup cell first."
        )
    resolved_repo = str(repo_path.resolve())
    if resolved_repo not in sys.path:
        sys.path.insert(0, resolved_repo)
    module = importlib.import_module("seed_vc_wrapper")
    model = module.SeedVCWrapper(device=device)
    if device.type == "mps":
        infer_pitch = model.rmvpe.infer_from_audio
        model.rmvpe.infer_from_audio = lambda *args, **kwargs: infer_pitch(
            *args, **kwargs
        ).astype("float32")
    return model


def convert_voice(
    model,
    source: Path,
    reference: Path,
    *,
    pitch_shift: int = 0,
    diffusion_steps: int = 30,
) -> np.ndarray:
    if not -12 <= pitch_shift <= 12:
        raise ValueError("Pitch shift must be between -12 and +12 semitones.")
    if not 5 <= diffusion_steps <= 100:
        raise ValueError("Diffusion steps must be between 5 and 100.")
    if not source.is_file() or not reference.is_file():
        raise FileNotFoundError("Both the separated vocal and voice reference must exist.")

    conversion = model.convert_voice(
        str(source),
        str(reference),
        diffusion_steps=diffusion_steps,
        f0_condition=True,
        auto_f0_adjust=False,
        pitch_shift=pitch_shift,
        stream_output=False,
    )
    while True:
        try:
            next(conversion)
        except StopIteration as result:
            audio = result.value
            break
    converted = np.asarray(audio, dtype="float32").squeeze()
    if converted.ndim != 1 or converted.size == 0 or not np.isfinite(converted).all():
        raise AudioPipelineError("Seed-VC returned empty or invalid audio.")
    return converted


def mix_tracks(
    converted_voice: np.ndarray,
    original_voice: np.ndarray,
    accompaniment: np.ndarray,
    *,
    voice_volume: float = 1.0,
) -> np.ndarray:
    if not 0 <= voice_volume <= 2:
        raise ValueError("Voice volume must be between 0 and 2.")
    voice = np.asarray(converted_voice, dtype="float32").reshape(-1)
    original = np.asarray(original_voice, dtype="float32").reshape(-1)
    music = np.asarray(accompaniment, dtype="float32")
    if music.ndim == 1:
        music = np.stack((music, music))
    elif music.ndim == 2 and music.shape[0] != 2 and music.shape[1] == 2:
        music = music.T
    if music.ndim != 2 or music.shape[0] != 2:
        raise ValueError("The accompaniment must be mono or stereo audio.")

    length = min(len(voice), len(original), music.shape[1])
    if length == 0:
        raise ValueError("The tracks do not contain any audio samples.")
    source_level = float(np.sqrt(np.mean(np.square(original[:length]))))
    converted_level = float(np.sqrt(np.mean(np.square(voice[:length]))))
    if source_level < 1e-8 or converted_level < 1e-8:
        raise ValueError("The source or converted vocal is silent.")

    gain = source_level / converted_level
    mixed = music[:, :length] + voice_volume * gain * voice[:length]
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.98:
        mixed *= 0.98 / peak
    return mixed


def normalize_mix(source: Path, destination: Path, *, target_lufs: int = -16) -> None:
    if not -24 <= target_lufs <= -12:
        raise ValueError("Final loudness must be between -24 and -12 LUFS.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-af",
            f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        "Normalizing the final mix",
    )
