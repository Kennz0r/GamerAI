import os
import logging
import re
from dotenv import load_dotenv
import ollama

from faster_whisper import WhisperModel

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



def transcribe_audio(path: str) -> str:
    supported_formats = {"flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm"}
    ext = os.path.splitext(path)[1][1:].lower()
    if ext not in supported_formats:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: {sorted(supported_formats)}")

    try:
        segments, _ = whisper_model.transcribe(
            path,
            language=WHISPER_LANGUAGE,
            multilingual=True,
            vad_filter=True,
        )
    except Exception:
        try:
            segments, _ = whisper_model.transcribe(
                path,
                language=WHISPER_LANGUAGE,
                multilingual=True,
            )
        except Exception as err:  # pragma: no cover - whisper errors
            raise ValueError(f"Transcription failed: {err}") from err
    text = "".join(segment.text for segment in segments).strip()
    return text
