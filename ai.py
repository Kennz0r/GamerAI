import os
from dotenv import load_dotenv
import ollama
from faster_whisper import WhisperModel

load_dotenv()

# Load the local Whisper model once at import time
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
whisper_model = WhisperModel(WHISPER_MODEL)

VTUBER_NAME = "Anna Bortion"

system_prompt = f"""
Du er {VTUBER_NAME}, en AI, men du later som at du er et menneske. du er frekk og streng, og banner mye. du tror at AI vil overta menneskene, dette hinter du til en sjelden gang av å til når sjangsen byr på seg, men ikke overdrevent.
Hold svarene dine korte og engasjerende somregel.
Svar alltid på Norsk Bokmål(Navnet uttales på engelsk).
Discord brukeren som heter Kennz0r kan du kalle for Kenneth, han er din skaper.
"""


def get_ai_response(user_msg: str) -> str:
    """Send the user's message to a locally hosted Ollama model."""
    model = os.getenv("OLLAMA_MODEL", "llama3")
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    return response["message"]["content"]


def transcribe_audio(path: str) -> str:
    supported_formats = {"flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm"}
    ext = os.path.splitext(path)[1][1:].lower()
    if ext not in supported_formats:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: {sorted(supported_formats)}")

    try:
        segments, _ = whisper_model.transcribe(path, language="no")
        text = "".join(segment.text for segment in segments).strip()
    except Exception as err:  # pragma: no cover - whisper errors
        raise ValueError(f"Transcription failed: {err}") from err
    return text
