"""Plantilla pública para configurar el cuaderno de extracción de audio."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PERSONAJE = "mi_proyecto"
VIDEOS: list[str] = []

DIR_VIDEO = RAIZ / "datos" / "video"
DIR_AUDIO = RAIZ / "datos" / "audio"
FICHA_VIDEOS = RAIZ / "datos" / "videos.json"

DIR_VIDEO.mkdir(parents=True, exist_ok=True)
DIR_AUDIO.mkdir(parents=True, exist_ok=True)
