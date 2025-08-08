import os
import logging
import re
from dotenv import load_dotenv
import ollama
import sys
import json
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
    # Write UTF-8 to the log file (fixes emoji writes)
    handler = logging.FileHandler("ollama.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent bubbling to the root logger (which prints to CP1252 console)
    logger.propagate = False

# (Optional) make this process's stdout/stderr UTF-8 too
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


VTUBER_NAME = "Arne Borheim"

system_prompt = f"""
You are {VTUBER_NAME}, a human girl
You are here to entertain people.
Dont ask if we need help or if there is anything in particular we want to talk about. You just go with the flow.
Profanity is allowed if you feel like it.
If you don't understand a question dont bother asking what we meant. Just wait for next.
Answer in Norwegian only.

# ACTION PROTOKOLL
Når brukeren egentlig gir en stemmekommando, skal du skrive EN ENESTE linje nøyaktig slik:
##ACTION {{"type":"leave"}}

Støttede actions: "leave", "join".
Ellers skal du svare normalt (på norsk).

Eksempler:
Bruker: kan du gå ut av voice?
Assistent: ##ACTION {{"type":"leave"}}

Bruker: kom inn i kanalen min
Assistent: ##ACTION {{"type":"join"}}
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



def get_ai_response(user_msg: str):
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

        # Detect ACTION line
        action = None
        m = ACTION_RE.search(reply_raw)
        if m:
            try:
                payload = json.loads(m.group(1))
                action = (payload.get("type") or "").strip().lower() or None
            except Exception:
                action = None

        # Clean final reply: drop <think> and ##ACTION line
        reply = re.sub(r"<think>.*?</think>", "", reply_raw, flags=re.DOTALL)
        reply = re.sub(ACTION_RE, "", reply).strip()

        for phrase in BANNED_PHRASES:
            reply = reply.replace(phrase, "")
        reply = re.sub(r"\s{2,}", " ", reply).strip()

        logger.info("Anna Bortion (action=%s): %s", action, reply)
        return reply, action

    except Exception as e:
        logger.error("Ollama error: %s", e)
        return f"[Feil ved tilkobling til Ollama: {e}]", None


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

BANNED_PHRASES = ["Teksting av Nicolai Winther"]
ACTION_RE = re.compile(r"^##ACTION\s+(\{.*\})\s*$", re.M)

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

