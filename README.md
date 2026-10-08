# Clonación y conversión de voz

Proyecto educativo para explorar tres flujos de trabajo locales con audio: preparar una pista, generar voz hablada a partir de una referencia y convertir el timbre de una voz cantada.

```text
Fuente de audio autorizada
          |
          +--> Descargar/extraer WAV -----------------------+
          |                                                  |
          +--> Texto + voz de referencia --> voz hablada    |
          |                                                  v
          +--> Canción --> voz / instrumental --> conversión + mezcla
```

Los cuadernos guardan las grabaciones y resultados en el equipo. No se suben al repositorio voces de referencia, canciones, vídeos, modelos ni resultados generados.

## Contenido

| Archivo | Para qué sirve |
| --- | --- |
| `01_descargar_video_extraer_audio.ipynb` | Consultar una fuente permitida y extraer una pista WAV mono a 16 kHz. También admite archivos locales. |
| `02_clonar_voz_chatterbox.ipynb` | Generar voz hablada a partir de texto y una muestra propia o autorizada. |
| `03_convertir_voz_cantada_seedvc.ipynb` | Separar voz e instrumental, convertir el timbre y mezclar el resultado. |
| `requirements-audio.txt` | Dependencias para el flujo de descarga y extracción. |
| `requirements-chatterbox.txt` | Dependencias del cuaderno Chatterbox. |
| `requirements-canto.txt` | Dependencias del flujo de Seed-VC y Demucs. |
| `config_example.py` | Configuración pública sin enlaces privados. |

## Requisitos

- Windows, macOS o Linux.
- Git para descargar Seed-VC desde su repositorio oficial.
- FFmpeg y `ffprobe` instalados y disponibles en `PATH` para descargar o convertir medios.
- Python 3.11 para Chatterbox; Python 3.10 para Seed-VC. Usa entornos virtuales separados: sus dependencias de PyTorch no se deben mezclar.
- Una GPU compatible acelera la conversión cantada. Sin GPU, usa fragmentos breves y reduce los pasos de inferencia.

## Puesta en marcha

### 1. Preparar audio (opcional)

Desde la raíz del repositorio crea un entorno e instala las dependencias:

```powershell
py -3.11 -m venv .venv-audio
.\.venv-audio\Scripts\Activate.ps1
python -m pip install -r requirements-audio.txt
```

Instala FFmpeg por separado y verifica que `ffmpeg` y `ffprobe` se puedan ejecutar desde una terminal. Abre `01_descargar_video_extraer_audio.ipynb` y ejecuta las celdas en orden.

La lista inicial de fuentes está vacía. Si vas a usar enlaces o archivos locales, crea `config.py` a partir de `config_example.py` y edítalo:

```powershell
Copy-Item config_example.py config.py
```

El archivo local `config.py` está ignorado por Git. Añade únicamente fuentes cuyo uso y descarga estén permitidos. Los resultados quedan en `datos/`.

### 2. Generar voz hablada con Chatterbox

En otra terminal, crea un entorno independiente:

```powershell
py -3.11 -m venv .venv-voz
.\.venv-voz\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-chatterbox.txt
python -m ipykernel install --user --name clonacion-voz
```

Abre `02_clonar_voz_chatterbox.ipynb` en VS Code con la extensión Jupyter y selecciona el kernel `clonacion-voz`. La primera carga descarga los pesos del modelo y requiere espacio y conexión. El cuaderno puede grabar con el micrófono o importar una referencia; los WAV se guardan en `audios/`.

### 3. Convertir voz cantada con Seed-VC

En una tercera terminal, usa Python 3.10 y otro entorno:

```powershell
py -3.10 -m venv .venv-canto
.\.venv-canto\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-canto.txt
python -m ipykernel install --user --name conversion-canto
```

Abre `03_convertir_voz_cantada_seedvc.ipynb` en VS Code con la extensión Jupyter y selecciona el kernel `conversion-canto`. Al ejecutar la celda de preparación, Seed-VC se descarga en `seed-vc/` si aún no está instalado; la primera inferencia también descarga pesos. Coloca una fuente autorizada en `datos/audio/cancion_entrada.wav` o cambia la variable `CANCION` por una ruta local. El cuaderno utiliza `audios/02_estilo_expresivo.wav` como referencia cuando está disponible y, si no, permite grabar una referencia.

## Datos y uso responsable

Las carpetas `datos/`, `audios/`, `audios_canto/`, los entornos y los modelos están ignorados por Git. No añadas grabaciones ni canciones a un commit. Usa tu propia voz o una voz para la que tengas consentimiento; respeta los derechos de las canciones y las condiciones de cada plataforma. Identifica el audio sintético como generado y no suplantes a otras personas.

Los PDF e imágenes del material de referencia no se redistribuyen aquí. La explicación y la estructura se han reescrito para este repositorio con autorización; el material de formación de base corresponde al Programa Momentum (CSIC) en colaboración con Upgrade Hub.

## Proyectos utilizados

- [Chatterbox](https://github.com/resemble-ai/chatterbox) y [Perth](https://github.com/resemble-ai/perth), de Resemble AI.
- [Seed-VC](https://github.com/Plachtaa/seed-vc).
- [Demucs](https://github.com/facebookresearch/demucs).
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) y [FFmpeg](https://ffmpeg.org/).

Consulta las licencias y condiciones vigentes de cada proyecto antes de redistribuir software, modelos o audio generado.
