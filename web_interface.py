import os, base64
import io
import subprocess
import sys
import signal
import json
import requests
from dotenv import load_dotenv
from flask import Flask, request, redirect, jsonify, send_from_directory
from ai import get_ai_response, transcribe_audio, set_model, ollama_client
import tempfile
import torch
import asyncio
import re, time
import regex as reg
from unidecode import unidecode
from pydub import AudioSegment, effects
import edge_tts
from flask import Response
import logging
import warnings
import torchaudio
import math
import wave
import time
from pydub import AudioSegment
from difflib import SequenceMatcher
from collections import deque, OrderedDict
import uuid, threading
from collections import defaultdict

load_dotenv()

torch.set_float32_matmul_precision("high")
logging.getLogger('werkzeug').setLevel(logging.WARNING)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # hush Transformers info/warns
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="The attention mask is not set")

if hasattr(torch, "load"):
    _orig_load = torch.load
    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)
    torch.load = _load
    print("[TTS] Patched torch.load to weights_only=False")

try:
    avail = torch.cuda.is_available()
    print(f"[CUDA] available={avail}")
    if avail:
        print(f"[CUDA] device={torch.cuda.get_device_name(0)}")
    else:
        print("[CUDA] GPU not available; using CPU.")
except Exception as e:
    print(f"[CUDA] query failed: {e}")

DISCORD_TEXT_CHANNEL = os.getenv("DISCORD_TEXT_CHANNEL", "0")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_API_BASE = "https://discord.com/api/v10"

def resolve_discord_name(user_id: str) -> str:
    if not user_id or not DISCORD_TOKEN:
        return user_id
    url = f"{DISCORD_API_BASE}/users/{user_id}"
    headers = {"Authorization": f"Bot {DISCORD_TOKEN}"}
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code == 200:
        return r.json().get("global_name") or r.json().get("username", user_id)
    return user_id

def _load_wav_best_effort(path: str) -> AudioSegment:
    """
    Prøver flere veier for å lese WAV-fila:
    1) AudioSegment.from_wav (ren wav-leser)
    2) AudioSegment.from_file(..., format='wav') (via ffmpeg)
    3) wave.open + AudioSegment.from_raw (manuelt)
    Returnerer AudioSegment (kan være 0ms hvis alt feiler).
    """
    try:
        seg = AudioSegment.from_wav(path)
        if len(seg) > 0:
            return seg
    except Exception as e:
        print(f"[srv] from_wav failed: {e}")

    try:
        seg = AudioSegment.from_file(path, format="wav")
        if len(seg) > 0:
            return seg
    except Exception as e:
        print(f"[srv] from_file(wav) failed: {e}")

    # Fallback: les header manuelt
    try:
        import wave as _wave
        with _wave.open(path, "rb") as wf:
            ch = wf.getnchannels()
            sr = wf.getframerate()
            sw = wf.getsampwidth()  # bytes per sample
            nf = wf.getnframes()
            raw = wf.readframes(nf)
            print(f"[srv] wave.open: ch={ch} sr={sr} sw={sw} frames={nf} bytes={len(raw)}")
        if nf > 0 and len(raw) > 0:
            return AudioSegment(
                data=raw,
                sample_width=sw,
                frame_rate=sr,
                channels=ch,
            )
    except Exception as e:
        print(f"[srv] wave.open failed: {e}")

    return AudioSegment.silent(duration=0)


# --- Edge-TTS config ---
VOICE_NAME = os.getenv("TTS_VOICE", "nb-NO-IselinNeural")  # or nb-NO-FinnNeural / en-US-AnaNeural
TTS_RATE  = os.getenv("TTS_RATE", "0%")   # slightly slower, more natural
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz") # neutral pitch
TTS_POSTPROCESS = os.getenv("TTS_POSTPROCESS", "true").lower() == "true"

IMAGE_USE_MODE = os.getenv("IMAGE_USE_MODE", "auto").lower()  # auto | always | never

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "piper").lower()

URL_RE = re.compile(r"https?://\S+")
CODE_FENCE = re.compile(r"```.*?```", re.S)
EMOJI_RE = reg.compile(r"\p{Emoji_Presentation}")

app = Flask(__name__)
pending = {"channel_id": None, "reply": None}
voice_command = {"action": None, "channel_id": None}
conversation = []
# Track whether speech recognition is enabled
speech_recognition_enabled = True
# Track whether sending to Discord is enabled
discord_send_enabled = True
# Pending TTS audio bytes for Discord bot and web preview
pending_tts_discord: bytes | None = None
# Pending preview audio is generated on-demand but keep storage for compatibility
pending_tts_web: bytes | None = None
# Handle for the optional Discord bot subprocess
discord_bot_process: subprocess.Popen | None = None
# Storage for optional fine-tuning examples

# Track processing durations in milliseconds
last_process_times = {"speech_ms": 0, "llm_ms": 0, "tts_ms": 0, "total_ms": 0}
LAST_TAIL: dict[str, AudioSegment] = {}
TAIL_MS = int(os.getenv("STT_TAIL_MS", "900"))  # overlap duration
STT_RECENT = {}  # channel_i
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "10"))
# --- STT post-processing helpers ---
_STT_RECENT = {}  # channel_id -> deque[(timestamp, normalized_text)]
# Visual intent patterns (used only when NO image is attached)

# én lås per bruker (hindrer race i LAST_TAIL/merging)
USER_LOCKS: dict[str, threading.Lock] = defaultdict(threading.Lock)

# dir for midlertidige lydfiler
os.makedirs("tmp_audio", exist_ok=True)

# 1) Øverst i filen (sammen med andre globale variabler)
VISION_CACHE = {}  # guild_id -> {"b64": str, "ts": float}
VISION_MAX_AGE = int(os.getenv("VISION_MAX_AGE", "45"))  # sekunder


# øverst sammen med VISION_CACHE
VISION_REQUESTS = []  # FIFO av {"channel_id": str, "guild_id": str, "ts": float}
VISION_AWAIT_MS = int(os.getenv("VISION_AWAIT_MS", "1200"))  # ventetid på web-push

def _vision_signal(channel_id: str | None, guild_id: str | None):
    if not channel_id and not guild_id:
        return
    VISION_REQUESTS.append({"channel_id": str(channel_id or ""), "guild_id": str(guild_id or ""), "ts": time.time()})

@app.route("/vision/request", methods=["GET"])
def vision_request():
    # dropp gamle requests (>5s)
    now = time.time()
    while VISION_REQUESTS and (now - VISION_REQUESTS[0]["ts"] > 5):
        VISION_REQUESTS.pop(0)
    if not VISION_REQUESTS:
        return jsonify({})
    req = VISION_REQUESTS.pop(0)
    return jsonify({"channel_id": req["channel_id"], "guild_id": req["guild_id"]})



def _vision_set(guild_id: str | None, img_b64: str | None):
    if not guild_id or not img_b64:
        return
    VISION_CACHE[guild_id] = {"b64": img_b64, "ts": time.time()}

def _vision_get(guild_id: str | None) -> str | None:
    if not guild_id:
        return None
    item = VISION_CACHE.get(guild_id)
    if not item:
        return None
    if (time.time() - item["ts"]) > VISION_MAX_AGE:
        return None
    return item["b64"]


VISUAL_PATTERNS = [
    # Norwegian
    r"\b(hva\s+er\s+dette|hva\s+ser\s+du|hva\s+ser\s+du\s+her|hva\s+er\s+det)\b",
    r"\b(på\s+skjermen|i\s+bildet|på\s+bildet)\b",
    r"\b(se|sjekk|kikk|ta\s+en\s+titt)\s*(på\s+)?(skjermen|bildet|screenshot|vindu|fane|kart)\b",
    r"\b(hvor\s+er\s+jeg|hvor\s+er\s+det)\b",
    # English
    r"\b(what\s+is\s+this|what\s+do\s+you\s+see|what\s+do\s+you\s+see\s+here|what\s+should\s+i\s+do\s+now|where\s+am\s+i)\b",
    r"\b(on\s+(the\s+)?screen|in\s+(the\s+)?image|in\s+(the\s+)?picture)\b",
]
VISUAL_SHORT_TRIGGERS = {"se", "se da", "se då", "look", "see", "check"}

def _needs_image(user_text: str) -> bool:
    """Only used when NO image is present: decide whether we should ask for one."""
    t = unidecode((user_text or "").lower()).strip()
    if not t:
        return False
    if t in VISUAL_SHORT_TRIGGERS:
        return True
    for pat in VISUAL_PATTERNS:
        if re.search(pat, t, flags=re.I):
            return True
    return False

def _should_use_image(user_text: str, image_b64: str | None) -> bool:
    """Decide whether to ATTACH the image to the model call."""
    has_img = bool(image_b64)
    mode = IMAGE_USE_MODE
    if mode == "never":
        return False
    if mode == "always":
        return has_img
    # mode == "auto": if there's an image, attach it; don't gate on text
    return has_img



def _stt_norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    # drop most punctuation (keep nordic letters)
    s = re.sub(r"[^\wæøåöäéèáàüçñ ]+", "", s)
    return s.strip()

def squash_stt(raw: str | list[str]) -> str:
    """
    Collapses multiple STT finals into one text:
    - removes blanks
    - removes near-duplicates that are basically the same line
    """
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.splitlines() if p.strip()]
    else:
        parts = [p.strip() for p in raw if isinstance(p, str) and p.strip()]
    if not parts:
        return ""

    out = []
    for p in parts:
        n = _stt_norm(p)
        if len(n) < 3:  # ignore too-short (e.g., "i", "og")
            continue
        if out:
            sim = SequenceMatcher(None, _stt_norm(out[-1]), n).ratio()
            if sim > 0.92:
                continue
        out.append(p)
    return " ".join(out).strip()

def stt_should_drop(channel_id: str | None, text: str,
                    min_chars: int = None, window_sec: int = None) -> bool:
    """
    Returns True if we should ignore this STT final:
    - too short (min_chars)
    - near-duplicate of something accepted in the last window_sec
    """
    min_chars = int(os.getenv("STT_MIN_CHARS", str(min_chars or 6)))
    window_sec = int(os.getenv("STT_DEDUPE_WINDOW", str(window_sec or 10)))

    n = _stt_norm(text)
    if len(n) < min_chars:
        return True

    key = str(channel_id or "default")
    dq = _STT_RECENT.setdefault(key, deque(maxlen=12))
    now = time.time()

    # drop if very similar to a recent accepted phrase
    for ts, prev in list(dq):
        if now - ts > window_sec:
            continue
        sim = SequenceMatcher(None, prev, n).ratio()
        if sim > 0.90:
            return True

    # accept: remember it
    dq.append((now, n))
    return False



def build_history_for_guild(guild_id: str | None, limit: int = HISTORY_TURNS) -> str:
    """Return compact recent turns for this guild only."""
    if not guild_id:
        return ""
    lines = []
    # only this guild, last N entries
    for c in [x for x in conversation if x.get("guild_id") == guild_id][-limit:]:
        u = c.get("user_name") or "Bruker"
        um = (c.get("user_message") or "").strip()
        ar = (c.get("reply") or "").strip()
        if um:
            lines.append(f"{u}: {um}")
        if ar:
            lines.append(f"AI: {ar}")
    return "\n".join(lines).strip()


def load_training_examples() -> list[list[dict[str, str]]]:
    """Load and convert training examples from JSONL."""
    examples: list[list[dict[str, str]]] = []
    if not os.path.exists("training_data.jsonl"):
        return examples
    with open("training_data.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append(
                [
                    {"role": "user", "content": record.get("prompt", "")},
                    {"role": "assistant", "content": record.get("response", "")},
                ]
            )
    return examples



training_data: list[list[dict[str, str]]] = load_training_examples()





def _normalize_text(txt: str) -> str:
    # remove code blocks & links
    txt = CODE_FENCE.sub("", txt)
    txt = URL_RE.sub("", txt)
    # strip emojis that TTS mangles
    txt = EMOJI_RE.sub("", txt)
    # common chat slang to speech
    subs = {
        r"\blol\b": "ha ha",
        r"\bomg\b": "å herregud",
        r"\bidk\b": "jeg vet ikke",
    }
    for k, v in subs.items():
        txt = re.sub(k, v, txt, flags=re.I)

    # ensure space after sentence enders
    txt = re.sub(r"([.!?])([A-ZÆØÅ])", r"\1 \2", txt)
    # soften ALL CAPS
    def _soft(m):
        w = m.group(0)
        return w.capitalize()
    txt = re.sub(r"\b[A-ZÆØÅ]{3,}\b", _soft, txt)
    return txt.strip()



def _ffmpeg_pitch_and_speed(in_wav: str, out_wav: str, semitones: float = 0.0, atempo: float = 1.0):
    """
    Pitch shift by N semitones and optionally change speed.
    Uses rubberband if available; otherwise falls back to asetrate+aresample+atempo.
    """
    pitch_factor = 2 ** (semitones / 12.0) if semitones else 1.0

    # 1) Try rubberband (your ffmpeg build shows --enable-librubberband)
    rb_filter = f"rubberband=pitch={pitch_factor:.6f}:tempo={atempo:.6f}"
    cmd = ["ffmpeg", "-y", "-i", in_wav, "-filter:a", rb_filter, out_wav]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0:
        return  # done

    # 2) Fallback: asetrate+aresample+atempo (needs numeric SR)
    with wave.open(in_wav, "rb") as wf:
        sr = wf.getframerate()

    # asetrate changes pitch & speed; compensate speed with atempo=1/pitch_factor,
    # and then apply user atempo on top
    total_atempo = (1.0 / pitch_factor) * (atempo if atempo else 1.0)

    def _chain_atempo(val: float) -> str:
        # make sure each atempo is in [0.5, 2.0]
        chain = []
        remaining = val
        while remaining < 0.5 or remaining > 2.0:
            step = 2.0 if remaining > 2.0 else 0.5
            chain.append(step)
            remaining /= step
        if abs(remaining - 1.0) > 1e-6:
            chain.append(remaining)
        return ",".join(f"atempo={x:.6f}" for x in chain)

    filters = [f"asetrate={int(sr * pitch_factor)}", f"aresample={sr}"]
    atempo_chain = _chain_atempo(total_atempo)
    if atempo_chain:
        filters.append(atempo_chain)
    afilter = ",".join(filters)

    cmd = ["ffmpeg", "-y", "-i", in_wav, "-filter:a", afilter, out_wav]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg pitch/speed failed (fallback): {res.stderr.decode('utf-8', errors='ignore')}")




def _tts_piper(text: str) -> bytes:
    exe = os.getenv("PIPER_EXE")
    voice = os.getenv("PIPER_VOICE")
    cfg = os.getenv("PIPER_VOICE_CFG")
    rate = os.getenv("PIPER_RATE", "1.0")
    pitch_st = float(os.getenv("PIPER_PITCH_ST", "0"))   # e.g. 3
    atempo = float(os.getenv("PIPER_ATEMPO", "1.0"))     # e.g. 1.08

    if not (exe and voice and cfg):
        raise RuntimeError("Piper not configured in .env")

    # 1) Prepare temp file for Piper output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpwav:
        raw_wav = tmpwav.name

    try:
        # 2) Run Piper → raw_wav
        cmd = [
        exe, "-m", voice, "-c", cfg, "-s", rate, "-f", raw_wav,
        "--length_scale", os.getenv("PIPER_LENGTH_SCALE", "0.95"),
        "--noise_scale", os.getenv("PIPER_NOISE_SCALE", "0.5"),
        "--noise_w",     os.getenv("PIPER_NOISE_W", "0.7"),
]
        p = subprocess.run(cmd, input=text.encode("utf-8"),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0 or not os.path.exists(raw_wav):
            raise RuntimeError(f"Piper failed: {p.stderr.decode('utf-8', errors='ignore')}")

        # 3) Optional pitch/speed shaping
        shaped_wav = raw_wav
        if pitch_st or (abs(atempo - 1.0) > 1e-6):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
                shaped_wav = tmp2.name
            _ffmpeg_pitch_and_speed(raw_wav, shaped_wav, semitones=pitch_st, atempo=atempo)

        # 4) Convert to mono, 16 kHz, MP3 for Discord/web
        seg = AudioSegment.from_file(shaped_wav, format="wav")
        seg = seg.set_channels(1).set_frame_rate(16000)
        out = io.BytesIO()
        seg.export(out, format="mp3", bitrate="96k")
        return out.getvalue()

    finally:
        # Cleanup
        try:
            os.remove(raw_wav)
        except:
            pass
        try:
            if 'shaped_wav' in locals() and shaped_wav != raw_wav:
                os.remove(shaped_wav)
        except:
            pass




def _split_sentences(txt: str):
    # conservative sentence split; break very long sentences on commas
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÆØÅ])", txt)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > 220:
            chunks = re.split(r",\s+", p)
            buf = ""
            for c in chunks:
                if len(buf) + len(c) < 180:
                    buf = (buf + ", " + c).strip(", ")
                else:
                    if buf: out.append(buf)
                    buf = c
            if buf: out.append(buf)
        else:
            out.append(p)
    return out

def _to_ssml(sentences, rate=TTS_RATE, pitch=TTS_PITCH):
    # small pause after short sentences, longer after long ones
    parts = []
    for s in sentences:
        dur_ms = min(800, max(160, int(len(s) * 4)))
        parts.append(f"<s>{s}</s><break time='{dur_ms}ms'/>")
    content = "".join(parts)
    # NB locale set by voice; content can stay Norwegian
    return f"""
<speak version='1.0' xml:lang='nb-NO'>
  <prosody rate='{rate}' pitch='{pitch}'>
    {content}
  </prosody>
</speak>
""".strip()

    
async def _synth_ssml(ssml: str, voice: str) -> bytes:
    # Collapse SSML -> plain text (your edge-tts build has no ssml=True)
    plain = re.sub(r"<[^>]+>", " ", ssml)
    plain = re.sub(r"\s+", " ", plain).strip()
    # Edge sometimes returns no audio for certain combos. Try progressively.
    def comm(v, rate=None, pitch=None):
        kwargs = dict(text=plain, voice=v)
        if rate and rate != "0%":
            kwargs["rate"] = rate
        if pitch and pitch not in ("0Hz","+0Hz","-0Hz"):
            kwargs["pitch"] = pitch
        return edge_tts.Communicate(**kwargs)

    alt_nb = "nb-NO-FinnNeural" if "Iselin" in voice else "nb-NO-IselinNeural"
    attempts = [
        (voice, None, None),
        (voice, "-6%", None),
        (voice, "-6%", "+0Hz"),
        (alt_nb, None, None),
        (alt_nb, "-6%", "+0Hz"),
        ("en-US-AnaNeural", None, None),      # diagnostic fallback
        ("en-US-AnaNeural", "-6%", "+0Hz"),
    ]

    last_err = None
    for v, r, p in attempts:
        try:
            print(f"[TTS] Trying v='{v}' rate='{r}' pitch='{p}'")
            buf, got = io.BytesIO(), False
            async for ch in comm(v, r, p).stream():
                if ch["type"] == "audio":
                    buf.write(ch["data"]); got = True
            if got and buf.tell() > 0:
                print(f"[TTS] Success v='{v}' rate='{r}' pitch='{p}' bytes={buf.tell()}")
                return buf.getvalue()
            else:
                print(f"[TTS] No audio v='{v}' rate='{r}' pitch='{p}'")
        except Exception as e:
            last_err = e
            print(f"[TTS] Attempt failed v='{v}' rate='{r}' pitch='{p}': {e}")
    print(f"[TTS] No audio after attempts. Last error: {last_err}")
    return b""



def _polish_audio(mp3_bytes: bytes) -> bytes:
    # normalize, trim a bit of head/tail silence so clips match loudness
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    trimmed = effects.strip_silence(audio, silence_len=150, silence_thresh=audio.dBFS-20)
    normalized = effects.normalize(trimmed)
    out = io.BytesIO()
    normalized.export(out, format="mp3", bitrate="128k")
    return out.getvalue()


# --- Small TTS cache (LRU) ---

_TTS_CACHE: "OrderedDict[str, bytes]" = OrderedDict()
TTS_CACHE_MAX = int(os.getenv("TTS_CACHE_MAX", "32"))

def _tts_cache_key(text: str) -> str:
    # Piper voice + tekst er nok når vi ikke bruker andre leverandører
    piper_v = os.getenv("PIPER_VOICE", "")
    return f"{piper_v}|{hash(text)}"

def _tts_cache_get(key: str) -> bytes | None:
    try:
        v = _TTS_CACHE.pop(key)   # bump LRU
        _TTS_CACHE[key] = v
        return v
    except KeyError:
        return None

def _tts_cache_put(key: str, value: bytes) -> None:
    _TTS_CACHE[key] = value
    while len(_TTS_CACHE) > TTS_CACHE_MAX:
        _TTS_CACHE.popitem(last=False)


def _edge_tts_sync(ssml: str, voice: str) -> bytes:
    """Kjør async Edge-TTS trygt fra sync-kontekst (Flask)."""
    try:
        import asyncio
        return asyncio.run(_synth_ssml(ssml, voice))
    except RuntimeError:
        # hvis en event-loop allerede kjører: bruk en separat tråd/loop
        result: dict[str, bytes] = {}
        import threading, asyncio as _aio
        def _runner():
            loop = _aio.new_event_loop()
            _aio.set_event_loop(loop)
            try:
                result["data"] = loop.run_until_complete(_synth_ssml(ssml, voice))
            finally:
                loop.close()
        t = threading.Thread(target=_runner, daemon=True)
        t.start(); t.join()
        return result.get("data", b"")


def create_tts_audio(text: str) -> bytes:
    try:
        clean = _normalize_text(text)[:1200]
        sentences = _split_sentences(clean) or [clean or ""]
        plain = ". ".join(sentences)

        # Alltid Piper (med Edge-fallback hvis Piper skulle feile)
        cache_key = _tts_cache_key(plain)
        cached = _tts_cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            audio = _tts_piper(plain)  # ← din eksisterende Piper-funksjon
        except Exception as piper_err:
            print("[TTS] Piper failed, trying Edge:", piper_err)
            ssml = _to_ssml(sentences)
            raw = _edge_tts_sync(ssml, VOICE_NAME)  # ← din eksisterende VOICE_NAME/Edge-path
            audio = _polish_audio(raw) if TTS_POSTPROCESS and raw else raw

        if not audio:
            return b""

        _tts_cache_put(cache_key, audio)
        return audio

    except Exception as e:
        print("[TTS] Error in create_tts_audio:", e)
        return b""




def send_to_discord(channel_id: str, text: str) -> None:
    """Send plain text message to Discord text channel."""
    url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages"
    headers = {"Authorization": f"Bot {DISCORD_TOKEN}"}
    payload = {"content": text}
    requests.post(url, headers=headers, json=payload, timeout=10)


@app.route("/image_policy", methods=["GET", "POST"])
def image_policy():
    global IMAGE_USE_MODE
    if request.method == "GET":
        return jsonify({"mode": IMAGE_USE_MODE})
    data = request.get_json(force=True) or {}
    mode = str(data.get("mode", "auto")).lower()
    if mode not in ("auto", "always", "never"):
        return jsonify({"error": "invalid mode"}), 400
    IMAGE_USE_MODE = mode
    return jsonify({"mode": IMAGE_USE_MODE})


@app.route("/queue", methods=["POST"])
def queue_message():
    data = request.get_json(force=True)
    start_total = time.time()
    default_channel = DISCORD_TEXT_CHANNEL if DISCORD_TEXT_CHANNEL != "0" else None
    channel_id = data.get("channel_id") or default_channel
    user_message = data.get("user_message", "")
    user_name = data.get("user_name", "Unknown")
    user_id = (data.get("user_id") or "").strip() or None
    guild_id = (data.get("guild_id") or "").strip() or None
    image_b64, img_present, img_src = _extract_image_from_request(request)
    if not img_present and (cached := _vision_get(guild_id)):
        image_b64 = cached
        img_present = True
        img_src = "cache"
    elif img_present:
        _vision_set(guild_id, image_b64)
    # ... etter _vision_get/_vision_set-blokka
# Hvis spørsmålet krever syn men vi mangler bilde → be web om å pushe
    if not img_present and _needs_image(user_message):
        _vision_signal(channel_id, guild_id)
    # vent et kort øyeblikk på at web pusher via /vision/update
        if guild_id and VISION_AWAIT_MS > 0:
            deadline = time.time() + (VISION_AWAIT_MS / 1000.0)
            while time.time() < deadline:
                cached = _vision_get(guild_id)
                if cached:
                    image_b64, img_present, img_src = cached, True, "webpush"
                    break
                time.sleep(0.05)

        
    history_text = build_history_for_guild(guild_id)
    prompt = (
        "Dette er en pågående samtale i en Discord-server.\n"
        + (f"Tidligere meldinger (kort):\n{history_text}\n\n" if history_text else "")
        + f"Nå sier {user_name}: {user_message}\n"
        "Svar naturlig på norsk og hold tråden i samtalen."
    )

    start = time.time()
    use_img = _should_use_image(user_message, image_b64)
    print(f"[IMG] policy: mode={IMAGE_USE_MODE}, img_present={img_present}, src={img_src}, will_use={use_img}")

    if use_img:
        prompt += "\n(Bare bruk bildet hvis jeg ba deg om det eller spørsmålet krever syn.)"
    reply_raw = get_ai_response(
        prompt,
        user_id=user_id,
        user_name=user_name,
        guild_id=guild_id,
        image=(f"data:image/*;base64,{image_b64}" if use_img and img_present else None),
    )
    last_process_times["llm_ms"] = int((time.time() - start) * 1000)
    if isinstance(reply_raw, tuple):
        reply, action = reply_raw
    else:
        reply, action = reply_raw, None
    if action == "leave":
       voice_command["action"] = "leave"
       voice_command["channel_id"] = data.get("channel_id")
       print(f"[Voice Command] Leave triggered by {user_name} (text)")
       last_process_times["tts_ms"] = 0
       last_process_times["total_ms"] = int((time.time() - start_total) * 1000)
       return {"status": "voice_command", "command": "leave"}

    conversation.append({
        "guild_id": guild_id,
        "channel_id": channel_id,
        "user_id": user_id,
        "user_name": user_name,
        "user_message": user_message,
        "reply": reply,
        "rating": None,
    })

    global pending_tts_web, pending_tts_discord
    start = time.time()
    audio = create_tts_audio(reply)
    last_process_times["tts_ms"] = int((time.time() - start) * 1000)
    last_process_times["speech_ms"] = 0
    last_process_times["total_ms"] = int((time.time() - start_total) * 1000)
    pending_tts_web = audio

    if discord_send_enabled:
        pending_tts_discord = audio
        if channel_id and DISCORD_TOKEN:
            send_to_discord(channel_id, reply)
        pending["channel_id"] = None
        pending["reply"] = None
        return {"status": "sent"}

    pending["channel_id"] = channel_id
    pending["reply"] = reply
    pending_tts_discord = None

    return {"status": "queued"}


@app.route("/queue_audio", methods=["POST"])
def queue_audio():
    user_message = ""
    default_channel = DISCORD_TEXT_CHANNEL if DISCORD_TEXT_CHANNEL != "0" else None
    channel_id = request.form.get("channel_id") or default_channel
    user_name = request.form.get("user_name", "Voice")
    user_id   = (request.form.get("user_id") or "").strip() or None
    guild_id  = (request.form.get("guild_id") or "").strip() or None
    audio_file = request.files.get("file")

    if not audio_file:
        return {"error": "no file"}, 400
    if not speech_recognition_enabled:
        return {"status": "ignored"}

    # --- unike stier pr request ---
    uid = uuid.uuid4().hex
    ext = os.path.splitext(audio_file.filename or "chunk.wav")[1] or ".wav"
    raw_path    = os.path.join("tmp_audio", f"{uid}{ext}")
    merged_path = os.path.join("tmp_audio", f"{uid}_merged.wav")

    audio_file.save(raw_path)

    # bilde-cache håndteres som før (men IKKE trigge _needs_image før vi har STT-tekst)
    image_b64, img_present, img_src = _extract_image_from_request(request)
    if not img_present and (cached := _vision_get(guild_id)):
        image_b64, img_present, img_src = cached, True, "cache"
    elif img_present:
        _vision_set(guild_id, image_b64)

    start_total = time.time()
    lock_key = user_id or "global"
    with USER_LOCKS[lock_key]:
        try:
            audio_seg = _load_wav_best_effort(raw_path)
            print(f"[srv] raw_loaded: dur_ms={len(audio_seg)} sr(guess)={getattr(audio_seg, 'frame_rate', 'NA')} ch={getattr(audio_seg, 'channels', 'NA')}")
            if len(audio_seg) == 0:
                # vi har faktisk tomt innhold – ingen vits å gå videre
                return {"status": "ignored"}

            # konverter til mono 16k og normaliser for STT
            audio_seg = audio_seg.set_channels(1).set_frame_rate(16000)
            audio_seg = effects.normalize(audio_seg)

            # tail-merge (beskyttert av lock)
            if user_id:
                prev_tail = LAST_TAIL.get(user_id)
                if prev_tail:
                    audio_seg = prev_tail + audio_seg
                LAST_TAIL[user_id] = audio_seg[-TAIL_MS:] if len(audio_seg) > TAIL_MS else audio_seg

            # skriv MERGED
            audio_seg.export(merged_path, format="wav")
            print(f"[srv] chunk uid={uid} dur_ms={len(audio_seg)} dBFS={getattr(audio_seg,'dBFS','NA')}")

            # STT på MERGED (din transcribe_audio lager ev. _dn-filer selv)
            start = time.time()
            user_message = transcribe_audio(merged_path)
            last_process_times["speech_ms"] = int((time.time() - start) * 1000)
            user_message = squash_stt(user_message)

            if stt_should_drop(channel_id, user_message, min_chars=6, window_sec=10):
                print(f"[STT] Suppressed: {user_message!r}")
                return {"status": "ignored"}

            # Nå har vi tekst → evt. be web om skjermbilde
            if not img_present and _needs_image(user_message):
                _vision_signal(channel_id, guild_id)
                if guild_id and VISION_AWAIT_MS > 0:
                    deadline = time.time() + (VISION_AWAIT_MS / 1000.0)
                    while time.time() < deadline:
                        cached = _vision_get(guild_id)
                        if cached:
                            image_b64, img_present, img_src = cached, True, "webpush"
                            break
                        time.sleep(0.05)

        except ValueError as err:
            return {"error": str(err)}, 400
        finally:
            # rydd unike filer
            for p in (raw_path, merged_path):
                try: os.path.exists(p) and os.remove(p)
                except: pass

            if not user_message.strip():
                return {"status": "ignored"}


    history_text = build_history_for_guild(guild_id)
    prompt = (
    "Dette er en pågående samtale i en Discord-server.\n"
    + (f"Tidligere meldinger (kort):\n{history_text}\n\n" if history_text else "")
    + f"Nå sier {user_name}: {user_message}\n"
    "Svar naturlig på norsk og hold tråden i samtalen."
)
    use_img = _should_use_image(user_message, image_b64)
    print(f"[IMG] policy: mode={IMAGE_USE_MODE}, img_present={img_present}, src={img_src}, will_use={use_img}")
    if use_img:
        prompt += "\n(Bare bruk bildet hvis jeg ba deg om det eller spørsmålet krever syn.)"

    start = time.time()
    reply_raw = get_ai_response(
    prompt,
    user_id=user_id,
    user_name=user_name,
    guild_id=guild_id,
    image=(f"data:image/*;base64,{image_b64}" if use_img and img_present else None),
)
    last_process_times["llm_ms"] = int((time.time() - start) * 1000)
    if isinstance(reply_raw, tuple):
        reply, action = reply_raw
    else:
        reply, action = reply_raw, None
        

    # Voice action handling (from voice → STT → LLM)
    if action == "leave":
        voice_command["action"] = "leave"
        voice_command["channel_id"] = channel_id  # bot will disconnect from any VC it’s in
        print(f"[Voice Command] Leave triggered by {user_name} (voice)")
        last_process_times["tts_ms"] = 0
        last_process_times["total_ms"] = int((time.time() - start_total) * 1000)
        return {"status": "voice_command", "command": "leave"}
    conversation.append({
        "guild_id": guild_id,
        "channel_id": channel_id,
        "user_id": user_id,
        "user_name": user_name,
        "user_message": user_message,
        "reply": reply,
        "rating": None,
    })

    global pending_tts_web, pending_tts_discord
    start = time.time()
    audio = create_tts_audio(reply)
    last_process_times["tts_ms"] = int((time.time() - start) * 1000)
    last_process_times["total_ms"] = int((time.time() - start_total) * 1000)
    pending_tts_web = audio

    if discord_send_enabled:
        pending_tts_discord = audio
        if channel_id and DISCORD_TOKEN:
            send_to_discord(channel_id, reply)
        pending["channel_id"] = None
        pending["reply"] = None
        return {"status": "sent"}

    pending["channel_id"] = channel_id
    pending["reply"] = reply
    pending_tts_discord = None

    return {"status": "queued"}

@app.route("/vision/update", methods=["POST"])
def vision_update():
    data = request.get_json(force=True) or {}
    guild_id = (data.get("guild_id") or "").strip() or None
    channel_id = (data.get("channel_id") or "").strip() or None
    img = data.get("image") or ""
    # Require a valid data URI
    if not (isinstance(img, str) and img.startswith("data:image/")):
        return {"error": "data:image/* nødvendig"}, 400

    # Resolve guild via channel if needed
    if not guild_id and channel_id and DISCORD_TOKEN:
        try:
            url = f"{DISCORD_API_BASE}/channels/{channel_id}"
            headers = {"Authorization": f"Bot {DISCORD_TOKEN}"}
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                guild_id = str(r.json().get("guild_id") or "")
        except Exception as e:
            print(f"[vision/update] resolve guild failed: {e}")

    if not guild_id:
        return {"error": "guild_id or channel_id required"}, 400

    img_b64 = img.split(",", 1)[1] if "," in img else img
    _vision_set(guild_id, img_b64)
    return {"status": "ok"}

@app.route("/", methods=["GET"])
def index():
    return send_from_directory("static", "index.html")


@app.route("/conversation", methods=["GET"])
def get_conversation():
    return jsonify(conversation)


@app.route("/pending_message", methods=["GET", "DELETE"])
def pending_message_route():
    if request.method == "GET":
        return jsonify({"reply": pending["reply"]})
    pending["channel_id"] = None
    pending["reply"] = None
    return jsonify({"status": "cleared"})


@app.route("/train", methods=["GET"])
def train_page():
    return send_from_directory("static", "train.html")


@app.route("/training_data", methods=["POST"])
def add_training_example():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    response = data.get("response", "")
    with open("training_data.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
    # Reload processed training data so it's ready for fine-tuning
    training_data.append([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ])
    return jsonify({"status": "added", "examples": len(training_data)})


@app.route("/rate", methods=["POST"])
def rate_prompt():
    data = request.get_json(force=True)
    index = data.get("index")
    rating = data.get("rating")
    if not isinstance(index, int) or index < 0 or index >= len(conversation):
        return jsonify({"error": "invalid index"}), 400
    if rating not in ("up", "down"):
        return jsonify({"error": "invalid rating"}), 400
    entry = conversation[index]
    entry["rating"] = rating
    if rating == "up":
        with open("training_data.jsonl", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "prompt": entry.get("user_message", ""),
                        "response": entry.get("reply", ""),
                    }
                )
                + "\n"
            )
        training_data.append(
            [
                {"role": "user", "content": entry.get("user_message", "")},
                {"role": "assistant", "content": entry.get("reply", "")},
            ]
        )
    else:
        # Drop negatively rated entries so they aren't used for future training
        conversation.pop(index)

    # Personality reinforcement (now inside the function)
    try:
        from ai import PERSONA, PERSONA_PATH
        s = PERSONA.get("style", {})
        delta = 0.05 if rating == "up" else -0.05
        for knob in ["humor", "empathy", "conciseness"]:
            s[knob] = float(min(1.0, max(0.0, float(s.get(knob, 0.5)) + delta)))
        PERSONA["style"] = s
        with open(PERSONA_PATH, "w", encoding="utf-8") as f:
            json.dump(PERSONA, f, ensure_ascii=False, indent=2)
    except Exception as _:
        pass

    return jsonify({"status": "ok", "rating": rating})



@app.route("/conversation_training", methods=["POST"])
def conversation_training():
    if not conversation:
        return jsonify({"status": "no_conversation", "examples": len(training_data)})
    count = 0
    with open("training_data.jsonl", "a", encoding="utf-8") as f:
        for c in conversation:
            f.write(json.dumps({"prompt": c.get("user_message", ""), "response": c.get("reply", "")}) + "\n")
            training_data.append([
                {"role": "user", "content": c.get("user_message", "")},
                {"role": "assistant", "content": c.get("reply", "")},
            ])
            count += 1
    return jsonify({"status": "added", "examples": len(training_data), "from_conversation": count})

@app.route("/fine_tune", methods=["POST"])
def fine_tune_model():
    if not training_data:
        return jsonify({"error": "no training data"}), 400

    base_model = os.getenv("OLLAMA_MODEL", "mistral")
    fine_tuned_model = f"{base_model}-ft"

    # Flatten messages from all training examples
    messages = [msg for pair in training_data for msg in pair]

    progress_updates: list[str] = []
    try:
        for progress in ollama_client.create(
            model=fine_tuned_model,
            from_=base_model,
            messages=messages,
            stream=True,
        ):
            if progress.status:
                progress_updates.append(progress.status)

        # Point future responses to the fine-tuned model
        set_model(fine_tuned_model)
        os.environ["OLLAMA_MODEL"] = fine_tuned_model

        return jsonify(
            {
                "status": "completed",
                "model": fine_tuned_model,
                "progress": progress_updates,
            }
        )
    except Exception as err:  # pragma: no cover - depends on external service
        return jsonify({"error": str(err)}), 500


@app.route("/log", methods=["GET"])
def get_log():
    try:
        with open("ollama.log", "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return jsonify({"log": "".join(lines[-200:])})
    except FileNotFoundError:
        return jsonify({"log": ""})


@app.route("/tts_audio", methods=["GET"])
def get_tts_audio():
    global pending_tts_discord
    wait_ms = int(request.args.get("wait_ms", "0"))
    if not pending_tts_discord and wait_ms > 0:
        deadline = time.time() + (wait_ms / 1000.0)
        while (pending_tts_discord is None) and time.time() < deadline:
            time.sleep(0.05)

    if pending_tts_discord:
        data = pending_tts_discord
        pending_tts_discord = None
        resp = Response(data, mimetype="audio/mpeg")
        resp.headers["Cache-Control"] = "no-store"
        return resp
    return ("", 204)



@app.route("/tts_preview", methods=["GET", "POST"])
def get_tts_preview():
    global pending_tts_web, VOICE_NAME, TTS_RATE, TTS_PITCH
    if request.method == "POST":
        data = request.get_json(force=True)
        text  = data.get("text", "")
        voice = data.get("voice") or VOICE_NAME
        rate  = data.get("rate")  or TTS_RATE
        pitch = data.get("pitch") or TTS_PITCH

        # Temporarily override for this call
        old_v, old_r, old_p = VOICE_NAME, TTS_RATE, TTS_PITCH
        VOICE_NAME, TTS_RATE, TTS_PITCH = voice, rate, pitch
        try:
            audio = create_tts_audio(text)
        finally:
            VOICE_NAME, TTS_RATE, TTS_PITCH = old_v, old_r, old_p

        pending["channel_id"] = None
        pending["reply"] = None
        resp = Response(audio, mimetype="audio/mpeg")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    if pending_tts_web:
        data = pending_tts_web
        pending_tts_web = None
        resp = Response(data, mimetype="audio/mpeg")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    return ("", 204)


@app.route("/speech_recognition", methods=["GET", "POST"])
def speech_recognition_route():
    global speech_recognition_enabled
    if request.method == "GET":
        return jsonify({"enabled": speech_recognition_enabled})
    data = request.get_json(force=True)
    speech_recognition_enabled = bool(data.get("enabled", True))
    return jsonify({"enabled": speech_recognition_enabled})

@app.route("/discord_send", methods=["GET", "POST"])
def discord_send_route():
    global discord_send_enabled
    if request.method == "GET":
        return jsonify({"enabled": discord_send_enabled})
    data = request.get_json(force=True) or {}
    val = data.get("enabled", True)
    if isinstance(val, str):
        val = val.lower() in ("true", "1", "yes", "on")
    discord_send_enabled = bool(val)
    return jsonify({"enabled": discord_send_enabled})


@app.route("/timings", methods=["GET"])
def timings_route():
    """Return timing information for last processed tasks."""
    return jsonify(last_process_times)


@app.route("/approve", methods=["POST"])
def approve():
    text = request.form.get("reply", "")
    pending_reply = text or pending["reply"] or ""
    channel_id = pending.get("channel_id") or (DISCORD_TEXT_CHANNEL if DISCORD_TOKEN else None)

    if conversation:
        conversation[-1]["reply"] = pending_reply

    # Generate TTS audio and store for playback
    global pending_tts_discord, pending_tts_web
    audio = create_tts_audio(pending_reply)
    pending_tts_discord = audio
    pending_tts_web = audio

    if channel_id and DISCORD_TOKEN and discord_send_enabled:
        send_to_discord(channel_id, pending_reply)

    pending["channel_id"] = None
    pending["reply"] = None
    return redirect(".")


@app.route("/voice", methods=["POST"])
def set_voice_command():
    action = request.form.get("action")
    channel_id = request.form.get("channel_id")
    voice_command["action"] = action
    voice_command["channel_id"] = channel_id
    return redirect(".")


@app.route("/voice", methods=["GET"])
def get_voice_command():
    wait_ms = int(request.args.get("wait_ms", "0"))
    if not voice_command["action"] and wait_ms > 0:
        deadline = time.time() + (wait_ms / 1000.0)
        while (not voice_command["action"]) and time.time() < deadline:
            time.sleep(0.05)
    if voice_command["action"]:
        result = {"action": voice_command["action"], "channel_id": voice_command["channel_id"]}
        voice_command["action"] = None
        voice_command["channel_id"] = None
        return jsonify(result)
    return jsonify({})


@app.route("/piper_settings", methods=["GET", "POST"])
def piper_settings():
    """Return or update Piper TTS voice settings."""
    mapping = {
        "voice": "PIPER_VOICE",
        "rate": "PIPER_RATE",
        "pitch_st": "PIPER_PITCH_ST",
        "atempo": "PIPER_ATEMPO",
        "length_scale": "PIPER_LENGTH_SCALE",
        "noise_scale": "PIPER_NOISE_SCALE",
        "noise_w": "PIPER_NOISE_W",
    }

    defaults = {
        "voice": "",
        "rate": "1.0",
        "pitch_st": "0",
        "atempo": "1.0",
        "length_scale": "0.95",
        "noise_scale": "0.5",
        "noise_w": "0.7",
    }

    if request.method == "POST":
        data = request.get_json(force=True)
        for key, env_key in mapping.items():
            if key in data:
                os.environ[env_key] = str(data[key])

    current = {k: os.getenv(env_key, defaults[k]) for k, env_key in mapping.items()}
    return jsonify(current)



@app.route("/persona", methods=["GET","POST"])
def persona_route():
    from ai import PERSONA, PERSONA_PATH, load_persona
    if request.method == "GET":
        return jsonify(PERSONA)
    data = request.get_json(force=True)
    try:
        PERSONA.update(data or {})
        with open(PERSONA_PATH, "w", encoding="utf-8") as f:
            json.dump(PERSONA, f, ensure_ascii=False, indent=2)
        load_persona()
        return jsonify({"status":"ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/discord_bot", methods=["POST"])
def control_discord_bot():
    global discord_bot_process
    if request.method == "GET":
        running = bool(discord_bot_process and discord_bot_process.poll() is None)
        return jsonify({"running": running})
    data = request.get_json(force=True)
    action = data.get("action")
    if action == "start":
        if not discord_bot_process or discord_bot_process.poll() is not None:
            discord_bot_process = subprocess.Popen([sys.executable, "discord_bot.py"])
            return jsonify({"status": "started"})
        return jsonify({"status": "already_running"})
    elif action == "stop":
        if discord_bot_process and discord_bot_process.poll() is None:
            if os.name == "nt":
                discord_bot_process.terminate()
            else:
                discord_bot_process.send_signal(signal.SIGINT)
            try:
                discord_bot_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                discord_bot_process.kill()
                discord_bot_process.wait()
            finally:
                discord_bot_process = None
            return jsonify({"status": "stopped"})
        return jsonify({"status": "not_running"})
    return jsonify({"error": "unknown action"}), 400

def _extract_image_from_request(req):
    """
    Returnerer (image_b64, img_present, src) der:
    - image_b64 er ren base64 (UTEN 'data:image/...;base64,')
    - img_present er True hvis vi har et gyldig, ikke-overstort bilde
    - src er 'json' eller 'form' eller 'none'
    """
    img = None
    src = "none"

    # JSON først
    if req.is_json:
        data = req.get_json(silent=True) or {}
        img = data.get("image")
        if img: src = "json"

    # Deretter form (queue_audio)
    if not img:
        img = req.form.get("image")
        if img: src = "form"

    # Må være data-URI av bilde
    if not img or not isinstance(img, str) or not img.startswith("data:image/"):
        return None, False, src

    # Plukk ut base64-delen
    if "," in img:
        img_b64 = img.split(",", 1)[1]
    else:
        img_b64 = img  # tåler ren base64 hvis du evt. sender det

    # Størrelsesgrense (samme for begge endepunkter)
    try:
        approx_bytes = (len(img_b64) * 3) // 4
        max_bytes = int(os.getenv("IMAGE_MAX_BYTES", "4000000"))  # 4 MB default
        if approx_bytes > max_bytes:
            print(f"[IMG] drop oversized: {approx_bytes} > {max_bytes}")
            return None, False, src
    except Exception as e:
        print(f"[IMG] size check failed: {e}")

    # Kjapp dekode-test for å unngå søppel som får modellen til å henge
    try:
        base64.b64decode(img_b64[:2000] + "==", validate=False)
    except Exception:
        print("[IMG] invalid base64 header sample")
        return None, False, src

    return img_b64, True, src

@app.route("/eval", methods=["POST"])
def eval_models():
    payload = request.get_json(force=True) or {}
    prompts = payload.get("prompts") or []
    gold    = payload.get("gold")
    active  = os.getenv("OLLAMA_MODEL", "mistral")
    model_b = payload.get("model_b") or active
    model_a = payload.get("model_a") or (active.split("-ft")[0] if "-ft" in active else active)
    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

    if not prompts:
        for pair in training_data[-10:]:
            for msg in pair:
                if msg.get("role") == "user":
                    prompts.append(msg.get("content",""))
        prompts = [p for p in prompts if p][:5]

    def _ask(model: str, prompt: str) -> str:
        try:
            resp = ollama_client.chat(
                model=model,
                keep_alive=keep_alive,
                messages=[
                    {"role": "system", "content": "Svar kort og presist på norsk."},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "512")),
                    "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "256")),
                    "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
                    "repeat_penalty": float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.05")),
                    "num_gpu": int(os.getenv("OLLAMA_NUM_GPU", "1")),
                },
            )
            return (resp.get("message", {}) or {}).get("content", "").strip()
        except Exception as e:
            return f"[error: {e}]"

    import re as _re
    def _overlap(a: str, b: str) -> float:
        aw = set(_re.findall(r"[\wøæåA-ZÆØÅ]+", (a or "").lower()))
        bw = set(_re.findall(r"[\wøæåA-ZÆØÅ]+", (b or "").lower()))
        if not aw or not bw:
            return 0.0
        inter = len(aw & bw)
        union = len(aw | bw)
        return round(inter / union, 4)

    results = []
    for i, p in enumerate(prompts):
        out_a = _ask(model_a, p)
        out_b = _ask(model_b, p)
        row = {"prompt": p, "model_a": model_a, "output_a": out_a, "model_b": model_b, "output_b": out_b}
        if gold and i < len(gold):
            row["gold"] = gold[i]
            row["score_a"] = _overlap(out_a, gold[i])
            row["score_b"] = _overlap(out_b, gold[i])
        results.append(row)

    return jsonify({"count": len(results), "results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False, use_reloader=False, threaded=True)


