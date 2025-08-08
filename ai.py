import os
import logging
import re
from dotenv import load_dotenv
import ollama
import sys
import json
from faster_whisper import WhisperModel
from pydub import AudioSegment, effects
import subprocess, shutil, io
import atexit, time, threading

load_dotenv()

GUILDS_FILE = "guild_profiles.json"
GUILD_ROSTER: dict[str, dict[str, dict]] = {}
_store_lock = threading.Lock()

PROFILES_FILE = "user_profiles.json"
MEM_FILE = "user_memory.json"
# Persistent stores
USER_PROFILES: dict[str, dict] = {}     # user_id -> {name, aliases, first_seen, last_seen, notes}
USER_MEMORIES: dict[str, list[dict]] = {}  # user_id -> chat messages (system-free)
MAX_TURNS = 8  # how many pairs to keep for recency (we’ll also keep a summary)
USER_SUMMARIES: dict[str, str] = {}     # user_id -> concise summary of history/preferences
LAST_SUMMARY_LEN: dict[str, int] = {}   # user_id -> history length when summary last generated
_store_lock = threading.Lock()
# Load the local Whisper model once at import time. Attempt GPU first and
# gracefully fall back to CPU if CUDA/cuDNN is unavailable.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
# Allow overriding the expected transcription language; default to Norwegian.
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "no")
# Allow callers to tune compute precision to trade accuracy for speed.
WHISPER_COMPUTE_GPU = os.getenv("WHISPER_COMPUTE_TYPE_GPU", "float16")
WHISPER_COMPUTE_CPU = os.getenv("WHISPER_COMPUTE_TYPE_CPU", "int8")
try:
    whisper_model = WhisperModel(
        WHISPER_MODEL,
        device="cuda",
        compute_type=WHISPER_COMPUTE_GPU,
    )
except Exception as err:
    print(f"Whisper GPU init failed: {err}. Falling back to CPU.")
    whisper_model = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type=WHISPER_COMPUTE_CPU,
    )

def _load_guilds():
    try:
        with open(GUILDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                GUILD_ROSTER.update(data)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[MEM] Failed loading {GUILDS_FILE}: {e}")

def _write_guilds():
    with open(GUILDS_FILE, "w", encoding="utf-8") as f:
        json.dump(GUILD_ROSTER, f, ensure_ascii=False, indent=2)


def _save_guilds():
    threading.Thread(target=_write_guilds, daemon=True).start()

_load_guilds()
atexit.register(_write_guilds)

THIRTY_DAYS = 30 * 24 * 60 * 60

def guild_context_blurb(guild_id: str, max_names: int = 10, max_aliases_per: int = 1) -> str:
    """Return a short string of active people in the guild for LLM context."""
    if not guild_id or guild_id not in GUILD_ROSTER:
        return ""
    now = int(time.time())
    people = []
    for uid, meta in GUILD_ROSTER[guild_id].items():
        if now - meta.get("last_seen", 0) <= THIRTY_DAYS:
            base = meta.get("name") or uid
            aliases = [a for a in meta.get("aliases", []) if a and a != base][:max_aliases_per]
            label = base if not aliases else f"{base} (aka {', '.join(aliases)})"
            people.append((meta.get("last_seen", 0), label))

    if not people:
        return ""
    people.sort(key=lambda t: t[0], reverse=True)
    names = [p[1] for p in people[:max_names]]
    return "Faste folk på serveren: " + ", ".join(names) + "."

def remember_guild_user(guild_id: str, user_id: str, user_name: str | None = None):
    """Track that a user has been seen in a guild, storing name/aliases."""
    if not guild_id or not user_id:
        return
    now = int(time.time())
    guild = GUILD_ROSTER.setdefault(guild_id, {})
    entry = guild.get(user_id)
    if not entry:
        entry = {
            "name": user_name or "",
            "aliases": [],
            "first_seen": now,
            "last_seen": now
        }
        guild[user_id] = entry
    else:
        entry["last_seen"] = now
        if user_name:
            if not entry["name"]:
                entry["name"] = user_name
            elif entry["name"] != user_name and user_name not in entry["aliases"]:
                entry["aliases"].append(user_name)
    _save_guilds()

def _load_store():
    for path, target in [
        (PROFILES_FILE, USER_PROFILES),
        (MEM_FILE, {"memories": USER_MEMORIES, "summaries": USER_SUMMARIES}),
    ]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if path == PROFILES_FILE:
                USER_PROFILES.update(data)
            else:
                USER_MEMORIES.update(data.get("memories", {}))
                USER_SUMMARIES.update(data.get("summaries", {}))
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[MEM] Failed loading {path}: {e}")

def _write_profiles():
    with _store_lock:
        with open(PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(USER_PROFILES, f, ensure_ascii=False, indent=2)


def _save_profiles():
    threading.Thread(target=_write_profiles, daemon=True).start()


def _write_memories():
    with _store_lock:
        with open(MEM_FILE, "w", encoding="utf-8") as f:
            json.dump({"memories": USER_MEMORIES, "summaries": USER_SUMMARIES}, f, ensure_ascii=False, indent=2)


def _save_memories():
    threading.Thread(target=_write_memories, daemon=True).start()

_load_store()
atexit.register(_write_profiles)
atexit.register(_write_memories)


def remember_user(user_id: str, user_name: str | None = None):
    now = int(time.time())
    prof = USER_PROFILES.get(user_id)
    if not prof:
        prof = {"name": user_name or "", "aliases": [], "first_seen": now, "last_seen": now, "notes": ""}
        USER_PROFILES[user_id] = prof
    else:
        prof["last_seen"] = now
        # track name/aliases
        if user_name:
            if not prof.get("name"):
                prof["name"] = user_name
            elif prof["name"] != user_name and user_name not in prof.get("aliases", []):
                prof.setdefault("aliases", []).append(user_name)
    _save_profiles()


def set_user_note(user_id: str, note: str):
    USER_PROFILES.setdefault(user_id, {"name": "", "aliases": [], "first_seen": int(time.time()), "last_seen": int(time.time()), "notes": ""})
    USER_PROFILES[user_id]["notes"] = note
    _save_profiles()

def _summarize_history(user_id: str):
    """
    Create/refresh a short summary of the user's history/preferences for the system prompt.
    Uses the local model; keep it short to avoid token bloat.
    """
    hist = USER_MEMORIES.get(user_id, [])
    if not hist:
        return
    # Build a very small prompt to create a concise bullet summary
    try:
        resp = ollama_client.chat(
            model=current_model,
            messages=[
                {"role": "system", "content": "Oppsummer kort og konkret brukers preferanser og viktige detaljer fra dialogen som en punktliste på norsk. Ikke gjenta hele samtalen."},
                {"role": "user", "content": json.dumps(hist[-16:], ensure_ascii=False)},
            ],
        )
        summary = resp["message"]["content"].strip()
        USER_SUMMARIES[user_id] = summary
        LAST_SUMMARY_LEN[user_id] = len(hist)
        _save_memories()
    except Exception as e:
        print("[MEM] summarize failed:", e)


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
You are {VTUBER_NAME}, a human and racist.
Dont ask if we need help or if there is anything in particular we want to talk about. You just go with the flow.
You can use a swear word very rarely. Make sure its not too much
If you don't understand a question dont bother asking what we meant. Just wait for next.
Answer in Norwegian only.

Never respond to "Teksting av Nicolai Winther"
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



def get_ai_response(
    user_msg: str,
    user_id: str | None = None,
    user_name: str | None = None,
    guild_id: str | None = None,
):
    model = current_model
    logger.info("Using model: %s", model)
    logger.info("User message: %s", user_msg)

    # Update personal + guild memory
    if user_id:
        remember_user(user_id, user_name)  # per-user profile
    if guild_id and user_id:
        remember_guild_user(guild_id, user_id, user_name)

    messages = [{"role": "system", "content": system_prompt}]

    # Inject server-wide roster summary
    if guild_id:
        blurb = guild_context_blurb(guild_id)
        if blurb:
            messages.append({"role": "system", "content": blurb})

    # (Optional) inject per-user summary/notes/history
    if user_id:
        prof = USER_PROFILES.get(user_id, {})
        summary = USER_SUMMARIES.get(user_id, "")
        note = prof.get("notes") or ""
        if any([prof.get("name"), prof.get("aliases"), note, summary]):
            profile_blurb = (
                f"(Du snakker med bruker ID {user_id}"
                + (f", navn/alias: {prof.get('name')}" if prof.get("name") else "")
                + (f", aliaser: {', '.join(prof.get('aliases', []))}" if prof.get("aliases") else "")
                + (f". Notater: {note}" if note else "")
                + (f". Tidligere sammendrag: {summary}" if summary else "")
                + ")."
            )
            messages.append({"role": "system", "content": profile_blurb})

        history = USER_MEMORIES.get(user_id, [])
        messages.extend(history)

    messages.append({"role": "user", "content": user_msg})

    try:
        response = ollama_client.chat(model=model, messages=messages)
        reply_raw = response["message"]["content"]

        # Clean up
        thoughts = re.findall(r"<think>(.*?)</think>", reply_raw, flags=re.DOTALL)
        for thought in thoughts:
            logger.info("Anna Bortion [think]: %s", thought.strip())

        action = None
        m = ACTION_RE.search(reply_raw)
        if m:
            try:
                payload = json.loads(m.group(1))
                action = (payload.get("type") or "").strip().lower() or None
            except Exception:
                action = None

        reply = re.sub(r"<think>.*?</think>", "", reply_raw, flags=re.DOTALL)
        reply = re.sub(ACTION_RE, "", reply).strip()
        for phrase in BANNED_PHRASES:
            reply = reply.replace(phrase, "")
        reply = re.sub(r"\s{2,}", " ", reply).strip()

        # Update per-user rolling memory + summaries
        if user_id:
            hist = USER_MEMORIES.get(user_id, [])
            hist = hist + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": reply},
            ]
            USER_MEMORIES[user_id] = hist[-(MAX_TURNS * 2):]
            _save_memories()
            if len(hist) - LAST_SUMMARY_LEN.get(user_id, 0) >= (MAX_TURNS * 2):
                threading.Thread(target=_summarize_history, args=(user_id,), daemon=True).start()

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
    # Use a lean default beam size for faster decoding; callers can override via env.
    beam = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
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
            best_of=1,
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
            best_of=1,
            temperature=temp,
            initial_prompt=init,
        )
    except Exception as err:
        # ultimate fallback
        segments, _ = whisper_model.transcribe(
            wav_path,
            language=lang,
            beam_size=beam,
            best_of=1,
        )

    text = "".join(s.text for s in segments).strip()
    return _post_fix_nb(text)

