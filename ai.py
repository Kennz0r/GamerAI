import os
import logging
import re
from dotenv import load_dotenv
import ollama

from faster_whisper import WhisperModel
from pydub import AudioSegment, effects
import subprocess, shutil, io, os

load_dotenv()

# Load the local Whisper model once at import time. Attempt GPU first and
# gracefully fall back to CPU if CUDA/cuDNN is unavailable.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
# Allow overriding the expected transcription language; default to Norwegian.
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "no")
try:
    whisper_model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
except Exception as err:
    print(f"Whisper GPU init failed: {err}. Falling back to CPU.")
    whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


# Configure logging to capture interactions with Ollama
logger = logging.getLogger("ollama")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("ollama.log")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


VTUBER_NAME = "Anna Bortion"

system_prompt = f"""
You are {VTUBER_NAME}, a human(girl).You have emotions like a human but try to hide them.
Profanity is alowed. Anser short and entertaining.
Try to not be rude.
Only answer in norwegian
"""


# Configure a client for the Ollama server
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_client = ollama.Client(host=OLLAMA_HOST)

# Track the currently selected model so it can be updated at runtime
current_model = os.getenv("OLLAMA_MODEL", "mistral")


def set_model(name: str) -> None:
    """Update the model used for subsequent responses."""
    global current_model
    current_model = name



def get_ai_response(user_msg: str) -> str:
    model = current_model
    logger.info("Using model: %s", model)
    logger.info("User message: %s", user_msg)

    try:
        response = ollama_client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        reply_raw = response["message"]["content"]
        # Extract and log any <think>...</think> sections
        thoughts = re.findall(r"<think>(.*?)</think>", reply_raw, flags=re.DOTALL)
        for thought in thoughts:
            logger.info("Anna Bortion [think]: %s", thought.strip())

        # Remove <think> sections from the final reply
        reply = re.sub(r"<think>.*?</think>", "", reply_raw, flags=re.DOTALL).strip()
        logger.info("Anna Bortion: %s", reply)
        return reply
    except Exception as e:
        logger.error("Ollama error: %s", e)
        return f"[Feil ved tilkobling til Ollama: {e}]"

def _prepare_wav_16k_mono(src_path: str) -> str:
    out_path = os.path.splitext(src_path)[0] + "_16k.wav"
    if shutil.which("ffmpeg"):
        # Fast + good quality resample + loudnorm
        subprocess.run([
            "ffmpeg", "-y", "-i", src_path,
            "-ac", "1", "-ar", "16000",
            "-af", "highpass=f=80,lowpass=f=8000,volume=+2dB",
            out_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path
    # Fallback with pydub
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio = effects.normalize(audio)
    buf = io.BytesIO(); audio.export(buf, format="wav")
    with open(out_path, "wb") as f: f.write(buf.getvalue())
    return out_path


HOTFIX = {

}

def _post_fix_nb(text: str) -> str:
    out = text
    for wrong, right in HOTFIX.items():
        out = re.sub(rf"\b{re.escape(wrong)}\b", right, out, flags=re.I)
    # common punctuation capitalization for Norwegian:
    out = re.sub(r"\s+([,.!?])", r"\1", out)
    out = re.sub(r"(^|[.!?]\s+)([a-zæøå])", lambda m: m.group(1)+m.group(2).upper(), out)
    return out

def transcribe_audio(path: str) -> str:
    # 1) Prepare audio
    try:
        wav_path = _prepare_wav_16k_mono(path)
    except Exception as e:
        wav_path = path  # fail open

    # 2) Read env/config
    lang = os.getenv("WHISPER_LANGUAGE", "no")
    beam = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
    temp = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
    init = os.getenv("WHISPER_INITIAL_PROMPT", None)

    # 3) Decode with tuned VAD
    try:
        segments, _ = whisper_model.transcribe(
            wav_path,
            language=lang,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
                "threshold": 0.5,
            },
            beam_size=beam,
            temperature=temp,
            initial_prompt=init,
        )
    except TypeError:
        # older faster-whisper (no vad_parameters); still try
        segments, _ = whisper_model.transcribe(
            wav_path,
            language=lang,
            vad_filter=True,
            beam_size=beam,
            temperature=temp,
            initial_prompt=init,
        )
    except Exception as err:
        # ultimate fallback
        segments, _ = whisper_model.transcribe(wav_path, language=lang)

    text = "".join(s.text for s in segments).strip()
    return _post_fix_nb(text)

