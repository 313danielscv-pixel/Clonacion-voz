from __future__ import annotations

import tempfile
from pathlib import Path

import librosa
import soundfile as sf
import streamlit as st
import torch

from audio_pipeline import (
    convert_voice,
    load_seedvc_model,
    mix_tracks,
    normalize_mix,
    prepare_source_audio,
    prepare_voice_reference,
    separate_vocal_stems,
)

ROOT = Path(__file__).resolve().parent
SEED_VC_PATH = ROOT / "seed-vc"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
PREVIEW_SECONDS = 120
SOURCE_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "aac", "mp4", "mkv", "webm"]
VOICE_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "aac"]

st.set_page_config(
    page_title="Lluvia de estrellas — estudio vocal",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def get_seedvc_model(device_name: str):
    return load_seedvc_model(SEED_VC_PATH, torch.device(device_name))


def _save_upload(upload, destination: Path) -> Path:
    suffix = Path(upload.name).suffix.lower()
    if not suffix:
        raise ValueError("The uploaded file must have a recognizable extension.")
    saved_path = destination.with_suffix(suffix)
    saved_path.write_bytes(upload.getvalue())
    return saved_path


def _process_mix(
    source_upload,
    voice_upload,
    *,
    pitch_shift: int,
    diffusion_steps: int,
    voice_volume: float,
    clean_reference: bool,
    normalize_final: bool,
    preview: bool = False,
) -> tuple[bytes, float]:
    device_name = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    with st.status(
        "Preparando la vista previa…" if preview else "Preparando la mezcla…",
        expanded=True,
    ) as status:
        with tempfile.TemporaryDirectory(prefix="lluvia_estrellas_") as temporary:
            work = Path(temporary)
            uploaded_source = _save_upload(source_upload, work / "source")
            uploaded_voice = _save_upload(voice_upload, work / "voice")

            source_wav = work / "prepared_source.wav"
            duration = prepare_source_audio(
                uploaded_source,
                source_wav,
                limit_seconds=PREVIEW_SECONDS if preview else None,
            )
            reference_wav = work / "reference.wav"
            prepare_voice_reference(
                uploaded_voice,
                reference_wav,
                clean=clean_reference,
            )
            status.update(label="Separando voz e instrumental…")
            vocal_path, accompaniment_path = separate_vocal_stems(
                source_wav, work / "separated"
            )
            original_voice, _ = librosa.load(vocal_path, sr=44_100, mono=True)
            accompaniment, _ = librosa.load(
                accompaniment_path, sr=44_100, mono=False
            )

            status.update(label=f"Convirtiendo la voz en {device_name.upper()}…")
            model = get_seedvc_model(device_name)
            converted = convert_voice(
                model,
                vocal_path,
                reference_wav,
                pitch_shift=pitch_shift,
                diffusion_steps=diffusion_steps,
            )
            mixed = mix_tracks(
                converted,
                original_voice,
                accompaniment,
                voice_volume=voice_volume,
            )
            mixed_path = work / "mix.wav"
            sf.write(mixed_path, mixed.T, 44_100, subtype="PCM_16")
            if normalize_final:
                final_path = work / "mix_normalized.wav"
                normalize_mix(mixed_path, final_path)
            else:
                final_path = mixed_path
            result = final_path.read_bytes()

        status.update(
            label="Vista previa lista" if preview else "Mezcla lista",
            state="complete",
            expanded=False,
        )
    return result, duration


st.markdown(
    """
    <style>
    :root {
        --ink: #fff8ff;
        --muted: #bdb3c8;
        --pink: #ff3d9f;
        --purple: #a34dff;
        --panel: rgba(12, 8, 19, .88);
    }
    .stApp {
        background:
            radial-gradient(ellipse at 50% -12%, rgba(154, 28, 215, .25), transparent 48%),
            radial-gradient(ellipse at 0% 38%, rgba(248, 25, 155, .14), transparent 34%),
            radial-gradient(ellipse at 100% 44%, rgba(124, 42, 255, .16), transparent 37%),
            #050309;
        color: var(--ink);
        font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    }
    header[data-testid="stHeader"] { background: #050309; }
    [data-testid="stAppViewContainer"] { background: transparent; }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        opacity: .68;
        background-image:
            radial-gradient(1px 1px at 12% 18%, rgba(255,255,255,.9) 50%, transparent 100%),
            radial-gradient(1px 1px at 76% 12%, rgba(255,190,255,.8) 50%, transparent 100%),
            radial-gradient(1px 1px at 44% 63%, rgba(255,255,255,.65) 50%, transparent 100%),
            radial-gradient(1.5px 1.5px at 90% 74%, rgba(226,160,255,.8) 50%, transparent 100%);
        background-size: 191px 173px, 257px 233px, 211px 227px, 313px 281px;
    }
    .main .block-container {
        position: relative;
        z-index: 1;
        max-width: 1120px;
        padding: 1.5rem 2rem 4rem;
    }
    h1, h2, h3 {
        font-family: ui-sans-serif, system-ui, sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -.035em;
    }
    .hero {
        position: relative;
        overflow: hidden;
        isolation: isolate;
        text-align: center;
        padding: 2.2rem 2rem 2.8rem;
        margin: .1rem 0 1rem;
        border: 1px solid rgba(207, 90, 255, .23);
        border-radius: 26px;
        background:
            radial-gradient(ellipse at 50% 110%, rgba(180, 32, 255, .22), transparent 57%),
            linear-gradient(145deg, rgba(17, 9, 24, .94), rgba(8, 6, 14, .86));
        box-shadow: 0 18px 64px rgba(0,0,0,.42), inset 0 1px rgba(255,255,255,.05);
    }
    .hero::before {
        content: "";
        position: absolute;
        z-index: -1;
        inset: auto -8% -55px;
        height: 130px;
        background: repeating-radial-gradient(
            ellipse at 50% 100%,
            transparent 0 18px,
            rgba(255, 54, 194, .2) 19px 20px,
            transparent 21px 35px
        );
        mask-image: linear-gradient(to top, #000, transparent 88%);
    }
    .eyebrow {
        color: #ff77c7;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .2em;
        text-transform: uppercase;
    }
    .hero h1 {
        display: inline-block;
        margin: .6rem 0;
        font-size: clamp(2.8rem, 7vw, 5rem);
        line-height: 1;
        letter-spacing: -.065em;
        background: linear-gradient(100deg, #fff 14%, #ff65b5 57%, #ad62ff 94%);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent !important;
    }
    .hero p { color: #c9bfd2; font-size: 1.05rem; max-width: 700px; margin: 0 auto; }
    .step-card {
        min-height: 100px;
        padding: 1rem 1.15rem;
        margin: .3rem 0 1rem;
        border-radius: 18px;
        border: 1px solid rgba(197, 83, 255, .25);
        background: linear-gradient(130deg, rgba(26, 12, 33, .86), rgba(12, 9, 18, .88));
        box-shadow: inset 0 1px rgba(255,255,255,.04), 0 8px 28px rgba(0,0,0,.18);
    }
    .step-number { color: #ff6abb; font-size: .76rem; font-weight: 700; letter-spacing: .13em; }
    .step-title { color: var(--ink); font: 650 1rem ui-sans-serif, system-ui, sans-serif; margin-top: .35rem; }
    .step-note { color: var(--muted); font-size: .84rem; margin-top: .15rem; }
    div[data-testid="stFileUploader"] section {
        border: 1px dashed rgba(174, 82, 255, .42);
        border-radius: 16px;
        background: rgba(9, 7, 15, .72);
        color: #ded3e7;
    }
    div[data-testid="stFileUploader"] { border-radius: 18px; }
    .stApp label { color: #e5d9eb !important; }
    div[data-testid="stFileUploader"] button {
        border: 1px solid rgba(204, 87, 255, .55);
        border-radius: 11px;
        color: #fff !important;
        background: linear-gradient(110deg, rgba(191, 37, 148, .92), rgba(111, 43, 188, .96));
    }
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #bdb3c8 !important;
    }
    .stApp [data-testid="stFileUploaderDropzoneInstructions"],
    .stApp [data-testid="stFileUploaderDropzoneInstructions"] * { color: #c9bdd2 !important; }
    div[data-testid="stAudioInput"] {
        padding: .4rem .65rem;
        border: 1px solid rgba(190, 80, 255, .34);
        border-radius: 16px;
        background: linear-gradient(115deg, rgba(31, 13, 39, .96), rgba(11, 8, 17, .98));
    }
    div[data-testid="stAudioInput"] > div:not([data-testid]) {
        border-radius: 12px;
        background: rgba(13, 8, 20, .94) !important;
    }
    div[data-testid="stAudioInput"] button {
        border: 1px solid rgba(255, 90, 187, .54);
        border-radius: 11px;
        color: white !important;
        background: linear-gradient(100deg, #e82c99, #8637e8);
    }
    div[data-testid="stAudioInput"] p,
    div[data-testid="stAudioInput"] span { color: #e6d9ed !important; }
    div[data-testid="stAudioInputWaveSurfer"] {
        border-radius: 11px;
        background: rgba(13, 8, 20, .94) !important;
    }
    div[data-testid="stAudioInputWaveSurfer"] * { background-color: transparent !important; }
    div[data-testid="stAudioInputWaveformTimeCode"] {
        color: #e6d9ed !important;
        background: transparent !important;
    }
    div.stButton > button[kind="primary"] {
        min-height: 3.2rem;
        border: 0;
        border-radius: 14px;
        color: white;
        background: linear-gradient(100deg, #ff348f, #aa29f5 72%, #7736ff);
        font: 700 1rem ui-sans-serif, system-ui, sans-serif;
        box-shadow: 0 8px 28px rgba(220, 42, 174, .25);
    }
    div.stButton > button[kind="primary"]:hover { border: 0; filter: brightness(1.12); }
    div.stButton > button[kind="secondary"], div.stButton > button:not([kind="primary"]) {
        border: 1px solid rgba(194, 91, 255, .42);
        border-radius: 14px;
        color: #f9eaff;
        background: linear-gradient(120deg, rgba(38, 18, 50, .76), rgba(16, 12, 24, .9));
    }
    div.stButton > button[kind="secondary"]:hover, div.stButton > button:not([kind="primary"]):hover {
        border-color: rgba(255, 83, 184, .78);
        color: white;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(190, 80, 255, .25);
        border-radius: 20px;
        background: rgba(10, 7, 16, .76);
    }
    div[data-testid="stExpander"] summary {
        border-radius: 20px;
        color: var(--ink) !important;
        background: rgba(12, 8, 19, .96) !important;
    }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: rgba(10, 7, 16, .76) !important;
    }
    input[type="radio"], input[type="checkbox"] { accent-color: #f23fae; }
    div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { color: #ff82ce !important; }
    audio { width: 100%; border-radius: 14px; }
    .privacy {
        color: #c8bfd2;
        border-left: 3px solid var(--pink);
        padding: .65rem 1rem;
        background: linear-gradient(95deg, rgba(238, 45, 163, .09), rgba(137, 43, 255, .04));
        border-radius: 0 12px 12px 0;
        font-size: .88rem;
    }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid rgba(190,80,255,.22);
        padding: 1rem;
        border-radius: 16px;
    }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Estudio Musical by DaVocalis</div>
      <h1>Lluvia de estrellas</h1>
      <p>Conviértete en el vocalista de tu canción favorita, entonando en la canción como hombre o mujer según la canción.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="privacy"><b>Procesamiento local:</b> esta app se ejecuta en tu ordenador. '
    'No se envían audios a servicios externos. Las copias de trabajo se borran al terminar; '
    'los archivos cargados y el resultado permanecen en la memoria de la sesión local hasta que expire.</div>',
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns(2, gap="large")
with left:
    st.markdown(
        '<div class="step-card"><div class="step-number">01 / PISTA</div>'
        '<div class="step-title">La canción o mezcla</div>'
        '<div class="step-note">Audio o vídeo con audio; el resultado será solo WAV.</div></div>',
        unsafe_allow_html=True,
    )
    source_upload = st.file_uploader(
        "Elige una pista",
        type=SOURCE_TYPES,
        help="Máximo 200 MB y 5 minutos. En vídeos solo se extrae el audio.",
        key="source_upload",
    )
with right:
    st.markdown(
        '<div class="step-card"><div class="step-number">02 / REFERENCIA</div>'
        '<div class="step-title">La voz que quieres usar</div>'
        '<div class="step-note">Una muestra clara; se procesa como máximo los primeros 25 segundos.</div></div>',
        unsafe_allow_html=True,
    )
    voice_upload = st.file_uploader(
        "Sube una grabación",
        type=VOICE_TYPES,
        help="Usa tu propia voz o una grabación para la que tengas permiso.",
        key="voice_upload",
    )
    recorded_voice = st.audio_input("O graba una muestra con el micrófono")
    voice_source = recorded_voice or voice_upload

preview_inputs = (
    source_upload.name,
    len(source_upload.getbuffer()),
    voice_source.name,
    len(voice_source.getbuffer()),
) if source_upload is not None and voice_source is not None else None
if "preview_inputs" in st.session_state and st.session_state["preview_inputs"] != preview_inputs:
    st.session_state.pop("preview_result", None)
    st.session_state.pop("preview_duration", None)
    st.session_state.pop("preview_settings", None)
    st.session_state.pop("preview_settings_label", None)
    st.session_state.pop("preview_inputs", None)

with st.expander("Ajustes de conversión y mezcla", expanded=True):
    pitch_choice = st.radio(
        "Afinando",
        ["✨ Manteniendo", "👨‍🎤 Como varón", "👩‍🎤 Como mujer"],
        horizontal=True,
        help="Manteniendo conserva el tono de la canción; las otras opciones lo bajan o elevan.",
    )
    pitch_shift = {
        "✨ Manteniendo": 0,
        "👨‍🎤 Como varón": -5,
        "👩‍🎤 Como mujer": 5,
    }[pitch_choice]
    st.caption(
        "Manteniendo respeta la altura de la voz en la canción; Como varón baja 5 semitonos "
        "y Como mujer la sube 5. El timbre objetivo sigue dependiendo de la voz de referencia."
    )

    option_left, option_middle, option_right = st.columns(3)
    with option_left:
        diffusion_steps = st.slider("Detalle de conversión", 10, 50, 30, 5)
        clean_reference = st.checkbox("Limpiar ruido de la referencia", value=True)
    with option_middle:
        voice_volume = st.slider("Volumen de la voz", 0.0, 2.0, 1.0, 0.05)
        normalize_final = st.checkbox("Igualar volumen final", value=True)
    with option_right:
        st.markdown("**Sobre el sonido**")
        st.caption(
            "La limpieza y la normalización ayudan con grabaciones bajas o ruidosas, "
            "pero no recuperan detalle que el micrófono no captó."
        )

st.markdown(
    '<div class="step-card"><div class="step-number">03 / VISTA PREVIA</div>'
    '<div class="step-title">Escucha antes de crear la mezcla</div>'
    '<div class="step-note">Procesa hasta los primeros 2 minutos con los ajustes que acabas de elegir.</div></div>',
    unsafe_allow_html=True,
)
preview_clicked = st.button(
    "Generar vista previa · hasta 2 minutos",
    key="generate_preview",
    type="primary",
    use_container_width=True,
)
if preview_clicked:
    st.session_state.pop("preview_result", None)
    st.session_state.pop("preview_duration", None)
    st.session_state.pop("preview_settings", None)
    st.session_state.pop("preview_settings_label", None)
    if source_upload is None or voice_source is None:
        st.error("Añade una pista y una grabación de voz para generar la vista previa.")
    elif len(source_upload.getbuffer()) > MAX_UPLOAD_BYTES or len(voice_source.getbuffer()) > MAX_UPLOAD_BYTES:
        st.error("Cada archivo debe ocupar como máximo 200 MB.")
    else:
        try:
            preview_bytes, preview_duration = _process_mix(
                source_upload,
                voice_source,
                pitch_shift=int(pitch_shift),
                diffusion_steps=diffusion_steps,
                voice_volume=voice_volume,
                clean_reference=clean_reference,
                normalize_final=normalize_final,
                preview=True,
            )
            st.session_state["preview_result"] = preview_bytes
            st.session_state["preview_duration"] = preview_duration
            st.session_state["preview_settings"] = (
                pitch_choice,
                diffusion_steps,
                voice_volume,
                clean_reference,
                normalize_final,
            )
            st.session_state["preview_settings_label"] = (
                f"{pitch_choice} · {diffusion_steps} pasos · voz {voice_volume:.2f}×"
            )
            st.session_state["preview_inputs"] = preview_inputs
            st.success("Previsualización lista con los ajustes seleccionados.")
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"No se pudo generar la vista previa: {exc}")

if "preview_result" in st.session_state:
    st.audio(st.session_state["preview_result"], format="audio/wav")
    st.caption(
        f"Previsualización de {st.session_state['preview_duration']:.0f} s · "
        f"{st.session_state['preview_settings_label']}"
    )
    if st.session_state["preview_settings"] != (
        pitch_choice,
        diffusion_steps,
        voice_volume,
        clean_reference,
        normalize_final,
    ):
        st.info("Cambiaste los ajustes. Genera otra vista previa para escuchar estos cambios.")

st.caption(
    "Usa grabaciones y canciones propias o autorizadas. La primera ejecución descarga los modelos "
    "de Seed-VC y Demucs si aún no están instalados."
)
generate = st.button("Crear mezcla", type="primary", use_container_width=True)


if generate:
    st.session_state.pop("mix_result", None)
    if source_upload is None or voice_source is None:
        st.error("Añade una pista y una grabación de voz antes de continuar.")
    elif len(source_upload.getbuffer()) > MAX_UPLOAD_BYTES or len(voice_source.getbuffer()) > MAX_UPLOAD_BYTES:
        st.error("Cada archivo debe ocupar como máximo 200 MB.")
    else:
        try:
            result_bytes, duration = _process_mix(
                source_upload,
                voice_source,
                pitch_shift=int(pitch_shift),
                diffusion_steps=diffusion_steps,
                voice_volume=voice_volume,
                clean_reference=clean_reference,
                normalize_final=normalize_final,
            )
            st.session_state["mix_result"] = result_bytes
            st.session_state["mix_duration"] = duration
            st.success("Tu mezcla está lista para escuchar y descargar.")
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"No se pudo crear la mezcla: {exc}")

if "mix_result" in st.session_state:
    st.subheader("Tu resultado")
    st.caption(f"Duración: {st.session_state['mix_duration'] / 60:.1f} min · WAV estéreo")
    st.audio(st.session_state["mix_result"], format="audio/wav")
    st.download_button(
        "Descargar WAV",
        data=st.session_state["mix_result"],
        file_name="lluvia_de_estrellas_mezcla.wav",
        mime="audio/wav",
        type="primary",
        use_container_width=True,
    )
