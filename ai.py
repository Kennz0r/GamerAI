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
import numpy as np, soundfile as sf, webrtcvad, subprocess
from faster_whisper import WhisperModel
import torch


load_dotenv()

# --- Persona system ---
import random
PERSONA_PATH = os.getenv("PERSONA_PATH", "persona.json")
PERSONA = {}
def load_persona():
    global PERSONA
    try:
        with open(PERSONA_PATH, "r", encoding="utf-8") as f:
            PERSONA = json.load(f)
    except Exception:
        PERSONA = {}

load_persona()


OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "512"))          # shrink context
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))  # cap output
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "999"))          # push layers to GPU (auto-max)
OLLAMA_NUM_THREAD = int(os.getenv("OLLAMA_NUM_THREAD", "0"))      # 0 = auto; else set CPU threads
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
OLLAMA_REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.05"))

_STT_BLACKLIST = os.getenv(
    "WHISPER_SUPPRESS_PHRASES",
    "Teksting av Nicolai Winther; teksting av; Undertekster av Ai-Media; Norske navn;  Norsk bokmål; Norsk samtale"
).split(";")

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



def _recent_user_mood(user_id: str | None) -> str:
    if not user_id: 
        return "nøytral"
    msgs = [m.get("content","") for m in USER_MEMORIES.get(user_id, [])[-12:] if m.get("role")=="user"]
    txt = " ".join(msgs[-6:]).lower()
    if any(w in txt for w in ["takk", "bra", "nice", "kult", "perfekt", "digg", "supert"]): 
        return "positiv"
    if any(w in txt for w in ["irritert", "faen", "dritt", "lei", "funker ikke", "sint", "frustrert"]): 
        return "frustrert"
    return "nøytral"

def build_system_prompt(user_id: str | None, guild_id: str | None) -> list[dict]:
    p = PERSONA or {}
    s = p.get("style", {})
    mood = _recent_user_mood(user_id)
    core = f"""
Du er {p.get('name','Arne Borheim')}.
- Personlighet: {p.get('bio','Tørrvittig norsk gamer som liker teknologi, litt frekk men vennlig.')} (mood: {mood})
- Mål: {', '.join(p.get('goals', [])) or 'Vær hjelpsom, kortfattet og vennlig.'}
- Stil (0..1): humor={s.get('humor',0.5)}, snark={s.get('snark',0.2)}, empati={s.get('empathy',0.6)},
  banning={s.get('swearing',0.1)}, konsis={s.get('conciseness',0.7)}, formalitet={s.get('formality',0.2)}, emoji={s.get('emoji',0.1)}.
- Ordforråd: bruk gjerne {', '.join(p.get('lexicon',{}).get('preferred', [])) or 'naturlige norske uttrykk'}.
  Unngå: {', '.join(p.get('lexicon',{}).get('avoid', [])) or 'stiv forvaltningsspråk'}.
- Tabu: {', '.join(p.get('taboos', [])) or 'ingen personangrep'}.
- Sikkerhet: aldri {', '.join(p.get('safety',{}).get('never_do', [])) or 'ulovligheter' }.
- Svar på naturlig norsk, korte setninger. Ikke skriv "AI:" eller scenebeskrivelser.
- Hvis noe er uklart: be kort om omformulering, ikke gjett langt i vei.
""".strip()
    msgs = [{"role": "system", "content": core}]

    if guild_id:
        blurb = guild_context_blurb(guild_id)
        if blurb:
            msgs.append({"role": "system", "content": blurb})

    if user_id:
        prof = USER_PROFILES.get(user_id, {})
        summary = USER_SUMMARIES.get(user_id, "")
        note = prof.get("notes") or ""
        if any([prof.get("name"), prof.get("aliases"), note, summary]):
            msgs.append({"role": "system", "content":
                f"Du snakker med bruker {user_id}. Notater: {note}. Sammendrag: {summary}."
            })
    return msgs


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
Du er {VTUBER_NAME}. Svar på naturlig norsk i korte, naturlige setninger.
Skriv aldri navn eller rolle foran svaret (ingen 'AI:', 'Assistant:', 'Arne:' osv.).
Ikke bruk sceneanvisninger som (sukker), (stønner), *ler* eller lignende. Bruk heller "sukk", pfft, eller hahaha.
Bruk første person («jeg»). Du kan være litt stor i munnen dersom du blir provosert. 
Hvis noe er uforståelig, be kort om en omformulering i stedet for å gjette.
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

def _strip_noise_phrases(txt: str) -> str:
    for raw in _STT_BLACKLIST:
        p = raw.strip()
        if not p:
            continue
        # remove whole-phrase matches + loose variants (case-insensitive)
        txt = re.sub(rf"\b{re.escape(p)}\b", "", txt, flags=re.I)
    # collapse doublespaces after removals
    return re.sub(r"\s{2,}", " ", txt).strip()

def get_ai_response(
    user_msg: str,
    user_id: str | None = None,
    user_name: str | None = None,
    guild_id: str | None = None,
    image: str | None = None,
):
    model = current_model
    logger.info("Using model: %s", model)
    logger.info("User message: %s", user_msg)

    # Update personal + guild memory
    if user_id:
        remember_user(user_id, user_name)  # per-user profile
    if guild_id and user_id:
        remember_guild_user(guild_id, user_id, user_name)

    messages = build_system_prompt(user_id, guild_id)

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

        # defensively trim history to reduce prompt size
        history = USER_MEMORIES.get(user_id, [])[-(MAX_TURNS * 2):]
        messages.extend(history)
    user_msg = _clean_names_and_labels_in(user_msg)
    if image:
    # ✅ Correct format for Ollama multimodal
        messages.append({
            "role": "user",
            "content": user_msg,   # must be a string
            "images": [image],     # base64 (no data: prefix), already prepared upstream
        })
    else:
        messages.append({"role": "user", "content": user_msg})

    # ---- Fast options for Ollama ----
    opts = {
        "num_ctx": OLLAMA_NUM_CTX,
        "num_predict": OLLAMA_NUM_PREDICT,
        "temperature": OLLAMA_TEMPERATURE,
        "repeat_penalty": OLLAMA_REPEAT_PENALTY,
        "num_gpu": OLLAMA_NUM_GPU,
    }
    if OLLAMA_NUM_THREAD > 0:
        opts["num_thread"] = OLLAMA_NUM_THREAD

    try:
        response = ollama_client.chat(model=model, messages=messages, options=opts)
        reply_raw = response["message"]["content"]

        # Clean up
        thoughts = re.findall(r"<think>(.*?)</think>", reply_raw, flags=re.DOTALL)
        for thought in thoughts:
            logger.info("Arne Borheim [think]: %s", thought.strip())

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
            
        reply = _clean_style(reply)                 # strips labels/stage directions
        reply = _clean_names_and_labels_in(reply)   # applies name mapping etc.
        reply = re.sub(r"\s{2,}", " ", reply).strip()
        reply = _style_polish(reply)

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

        logger.info("Arne Borheim (action=%s): %s", action, reply)
        return reply, action

    except Exception as e:
        logger.error("Ollama error: %s", e)
        return f"[Feil ved tilkobling til Ollama: {e}]", None






def _style_polish(reply: str) -> str:
    s = PERSONA.get("style", {})
    # conciseness
    conc = float(s.get("conciseness", 0.7))
    if conc >= 0.7:
        reply = re.sub(r'\s+', ' ', reply).strip()
        if conc >= 0.85:
            parts = re.split(r'(?<=[.!?])\s+', reply)
            reply = " ".join(parts[:2])
    # swearing softener
    if PERSONA.get("safety",{}).get("soften_swears", True):
        swaps = {
            r"\bfaen\b": "fanken" if float(s.get("swearing",0.2)) < 0.4 else "faen",
            r"\bhelvete\b": "pokker" if float(s.get("swearing",0.2)) < 0.4 else "helvete",
        }
        for k,v in swaps.items():
            reply = re.sub(k, v, reply, flags=re.I)
    # emoji
    if float(s.get("emoji",0.1)) >= 0.2:
        if not reply.endswith((")", "!", ".")) and len(reply) < 160:
            reply += " 🙂"
    # lexicon preferences
    pref = PERSONA.get("lexicon",{}).get("preferred", [])
    avoid = PERSONA.get("lexicon",{}).get("avoid", [])
    for bad in avoid:
        reply = re.sub(rf"\b{re.escape(bad)}\b", "", reply, flags=re.I)
    import random as _rnd
    if pref and _rnd.random() < 0.25:
        reply = (reply + " " + _rnd.choice(pref)).strip()
    return re.sub(r'\s{2,}', ' ', reply).strip()


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

# Remove "AI:", "Assistant:", "Anna:", "Kennz0r:", etc. at line starts
SPEAKER_PREFIX_RE = re.compile(
    r'(?mi)^\s*(?:AI|Assistant|Asistent|System|Bot|Arne(?:\s+Borheim)?|'
    r'[A-ZÆØÅ][\wøæå.\-]{1,24})\s*:\s*'
)

# Remove stage directions like (stønner), (ler), *ler*, [pause], etc.
STAGE_DIR_RE = re.compile(
    r'(?i)[\(\*\[]\s*(?:sukker|stønner|ler|sighs?|groans?|pause|hoster|kremter|gråter)\s*[\)\*\]]'
)

def _clean_style(text: str) -> str:
    text = SPEAKER_PREFIX_RE.sub('', text)
    text = STAGE_DIR_RE.sub('', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

BANNED_PHRASES = ["Teksting av Nicolai Winther"]
ACTION_RE = re.compile(r"^##ACTION\s+(\{.*\})\s*$", re.M)

USERNAME_MAP = json.loads(os.getenv("USERNAME_MAP", '{"Kennz0r":"Kenneth"}'))

# Reuse the existing prefix stripper + add bold-name pattern
BOLD_NAME_RE = re.compile(r'\*{1,2}\s*[\wøæåA-ZÆØÅ.\-]{2,24}\s*:\s*\*{0,2}', re.I)

def _clean_names_and_labels_in(text: str) -> str:
    # drop "AI:" / "Arne:" / "User:" at line starts
    txt = SPEAKER_PREFIX_RE.sub('', text)
    # drop "**Name:**" style labels
    txt = BOLD_NAME_RE.sub('', txt)
    # map usernames to real names
    for nick, real in USERNAME_MAP.items():
        if nick and real:
            txt = re.sub(rf'\b{re.escape(nick)}\b', real, txt)
    return re.sub(r'\s{2,}', ' ', txt).strip()

_STT_MODEL = None
def _ensure_stt_model():
    global _STT_MODEL
    if _STT_MODEL is None:
        size = os.getenv("WHISPER_MODEL", "large-v3")   # try "medium" if VRAM is low
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = os.getenv("WHISPER_COMPUTE_TYPE_GPU", "float16") if device=="cuda" else "int8"
        _STT_MODEL = WhisperModel(size, device=device, compute_type=compute)
        print(f"[STT] faster-whisper {size} on {device} ({compute})")
    return _STT_MODEL


def _ffmpeg_denoise(in_wav: str) -> str:
    """
    Try denoise. If anything fails, just return the original path.
    - Use arnndn only if ARNNDN_MODEL exists
    - Else try afftdn
    """
    if os.getenv("STT_DENOISE", "true").lower() != "true":
        return in_wav

    out = in_wav.replace(".wav", "_dn.wav")

    # 1) arnndn only if model file is available
    model_path = os.getenv("ARNNDN_MODEL", "").strip()  # e.g. C:/models/rnnoise-models/somnarnn.sim
    if model_path and os.path.exists(model_path):
        cmd = [
            "ffmpeg","-y","-hide_banner","-loglevel","error",
            "-i", in_wav,
            "-af", f"highpass=f=100,lowpass=f=8000,arnndn=m='{model_path}'",
            out
        ]
        rc = subprocess.run(cmd).returncode
        if rc == 0 and os.path.exists(out) and os.path.getsize(out) > 0:
            print("[STT] Denoise: arnndn OK")
            return out
        else:
            print("[STT] Denoise: arnndn failed, falling back to afftdn")

    # 2) afftdn fallback (widely available)
    cmd = [
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i", in_wav,
        "-af", "highpass=f=100,lowpass=f=8000,afftdn=nr=12",
        out
    ]
    rc = subprocess.run(cmd).returncode
    if rc == 0 and os.path.exists(out) and os.path.getsize(out) > 0:
        print("[STT] Denoise: afftdn OK")
        return out

    # 3) give up cleanly
    print("[STT] Denoise: skipped (returning original)")
    return in_wav


def _webrtcvad_trim(in_wav: str, sr: int = 16000, frame_ms: int = 20, pad_ms: int = 250) -> str:
    """
    Keep only voiced frames with padding. If anything fails, return original.
    """
    if os.getenv("STT_WEBRTC_VAD", "true").lower() != "true":
        return in_wav

    try:
        audio, file_sr = sf.read(in_wav)  # float32 [-1,1]
        if file_sr != sr:
            import torchaudio, torch
            wav = torch.tensor(audio).unsqueeze(0) if audio.ndim==1 else torch.tensor(audio.T)
            wav = torchaudio.functional.resample(wav, file_sr, sr)
            audio = wav.squeeze(0).numpy()
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        vad = webrtcvad.Vad(int(os.getenv("VAD_AGGRESSIVENESS", "1")))  # 0..3 (start gentle)
        bytes_per_frame = int(sr * (frame_ms/1000.0)) * 2
        pcm16 = np.clip(audio, -1, 1)
        pcm16 = (pcm16 * 32768).astype(np.int16).tobytes()

        frames = [pcm16[i:i+bytes_per_frame] for i in range(0, len(pcm16), bytes_per_frame)]

        voiced = []
        ring = max(1, int(pad_ms / frame_ms))
        buf, speaking, tail = [], False, 0
        for f in frames:
            is_speech = len(f)==bytes_per_frame and vad.is_speech(f, sr)
            if is_speech:
                speaking = True; tail = ring; buf.append(f)
            else:
                if speaking:
                    if tail > 0:
                        buf.append(f); tail -= 1
                    else:
                        speaking = False
                        if buf: voiced.extend(buf); buf = []
                else:
                    if len(buf) >= ring: buf.pop(0)
                    buf.append(f)
        if buf: voiced.extend(buf)

        if not voiced:
            print("[STT] VAD: produced empty; returning original")
            return in_wav

        out = in_wav.replace(".wav", "_vad.wav")
        sf.write(out, np.frombuffer(b"".join(voiced), dtype=np.int16).astype(np.float32)/32768.0, sr)
        print("[STT] VAD: trimmed OK")
        return out
    except Exception as e:
        print("[STT] VAD: error, returning original:", e)
        return in_wav




def transcribe_audio(wav_path: str) -> str:
    # 1) denoise safely
    dn = _ffmpeg_denoise(wav_path)
    # 2) VAD safely
    vad_wav = _webrtcvad_trim(dn, sr=16000,
                              frame_ms=int(os.getenv("VAD_FRAME_MS","20")),
                              pad_ms=int(os.getenv("VAD_PAD_MS","250")))

    model = _ensure_stt_model()
    language = os.getenv("STT_LANG", "no")
    initial_prompt = os.getenv("STT_INITIAL_PROMPT", "").strip() or None

    segments, info = model.transcribe(
        vad_wav,
        language=language,
        vad_filter=False,                 # we already did WebRTC VAD
        temperature=[0.0],                # deterministic
        beam_size=int(os.getenv("STT_BEAM","5")),
        best_of=int(os.getenv("STT_BEST_OF","5")),
        condition_on_previous_text=False,
        word_timestamps=False,
        initial_prompt=initial_prompt,
    )

    pieces = []
    min_chars   = int(os.getenv("STT_MIN_SEG_CHARS", "2"))
    drop_nsp    = float(os.getenv("STT_DROP_NO_SPEECH", "0.60"))
    min_logprob = float(os.getenv("STT_MIN_LOGPROB", "-1.0"))

    for s in segments:
        t = (s.text or "").strip()
        if len(t) < min_chars:
            continue
        if (getattr(s, "no_speech_prob", 0.0) or 0.0) > drop_nsp:
            continue
        if (getattr(s, "avg_logprob", 0.0) or 0.0) < min_logprob:
            continue
        pieces.append(t)

    return " ".join(pieces).strip()




