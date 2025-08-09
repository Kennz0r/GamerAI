import os
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
import asyncio, re
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

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

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

# --- Edge-TTS config ---
VOICE_NAME = os.getenv("TTS_VOICE", "nb-NO-IselinNeural")  # or nb-NO-FinnNeural / en-US-AnaNeural
TTS_RATE  = os.getenv("TTS_RATE", "0%")   # slightly slower, more natural
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz") # neutral pitch
TTS_POSTPROCESS = os.getenv("TTS_POSTPROCESS", "true").lower() == "true"

# --- ElevenLabs config ---
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "coqui").lower()
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "").strip()
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "").strip()
ELEVEN_VOICE_NAME = os.getenv("ELEVEN_VOICE_NAME", "").strip()
ELEVEN_MODEL_ID = os.getenv("ELEVEN_MODEL_ID", "eleven_multilingual_v2")

# --- Coqui TTS config ---
COQUI_MODEL = os.getenv("COQUI_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
COQUI_LANGUAGE = os.getenv("COQUI_LANGUAGE", "no")
COQUI_SPEAKER_WAV = os.getenv("COQUI_SPEAKER_WAV", "").strip()


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

HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "10"))

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


_coqui_model = None


def _coqui_ensure_model():
    global _coqui_model
    if _coqui_model is not None:
        return _coqui_model
    from TTS.api import TTS
    import torch

    print(f"[TTS] Loading Coqui model: {COQUI_MODEL}")
    _coqui_model = TTS(model_name=COQUI_MODEL)

    if torch.cuda.is_available():
        print("[TTS] Moving model to GPU...")
        _coqui_model.to("cuda")
    else:
        print("[TTS] GPU not available, using CPU.")

    # --- PATCH tokenizer to always give attention_mask ---
    try:
        tok = getattr(_coqui_model, "tokenizer", None)
        if tok:
            orig_call = tok.__call__
            def call_with_mask(*args, **kwargs):
                kwargs.setdefault("return_attention_mask", True)
                return orig_call(*args, **kwargs)
            tok.__call__ = call_with_mask
            print("[Patch] Coqui tokenizer now always returns attention_mask.")
    except Exception as e:
        print("[Patch] Could not patch tokenizer:", e)

    return _coqui_model

def _tts_coqui(text: str) -> bytes:
    model = _coqui_ensure_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        base = {"text": text, "file_path": tmp_path}

        if COQUI_SPEAKER_WAV and os.path.exists(COQUI_SPEAKER_WAV):
            base["speaker_wav"] = COQUI_SPEAKER_WAV
        else:
            raise RuntimeError("Coqui XTTS needs a speaker_wav. Set COQUI_SPEAKER_WAV in .env.")

        # Use English (you said no mapping)
        base["language"] = "en"

        # 1) Synthesize at model's native sample rate
        model.tts_to_file(**base)

        # 2) Load the wav, resample to 16 kHz properly, then MP3 encode
        wav, sr = torchaudio.load(tmp_path)       # wav shape: [channels, samples]
        target_sr = 16000
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)

        # Convert to mono if needed
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Save a temp resampled wav and export to mp3 via pydub/ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as rs_wav:
            rs_path = rs_wav.name
        try:
            torchaudio.save(rs_path, wav, target_sr)  # proper resampled WAV
            seg = AudioSegment.from_file(rs_path, format="wav")
            out = io.BytesIO()
            seg.export(out, format="mp3", bitrate="64k")
            return out.getvalue()
        finally:
            try: os.remove(rs_path)
            except: pass

    finally:
        try: os.remove(tmp_path)
        except: pass

def _eleven_fetch_voice(voice_id: str) -> dict:
    r = requests.get(
        f"https://api.elevenlabs.io/v1/voices/{voice_id}",
        headers={"xi-api-key": ELEVEN_API_KEY},
        timeout=15,
    )
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"ElevenLabs voice lookup failed ({r.status_code}): {err}")
    return r.json()

def _eleven_resolve_voice_id() -> str:
    if ELEVEN_VOICE_ID:
        # Verify access and log the name to avoid surprises
        meta = _eleven_fetch_voice(ELEVEN_VOICE_ID)
        print(f"[TTS] Using Eleven voice ID {ELEVEN_VOICE_ID} -> name='{meta.get('name')}'")
        return ELEVEN_VOICE_ID

    if not ELEVEN_VOICE_NAME:
        raise RuntimeError("Set ELEVEN_VOICE_ID or ELEVEN_VOICE_NAME in .env")

    # No ID given: lookup by name (fresh; no cache)
    r = requests.get(
        "https://api.elevenlabs.io/v1/voices",
        headers={"xi-api-key": ELEVEN_API_KEY},
        timeout=15,
    )
    r.raise_for_status()
    voices = r.json().get("voices", [])
    cand = [v for v in voices if v.get("name","").lower() == ELEVEN_VOICE_NAME.lower()]
    if not cand:
        cand = [v for v in voices if ELEVEN_VOICE_NAME.lower() in v.get("name","").lower()]
    if not cand:
        raise RuntimeError(f"ElevenLabs voice named '{ELEVEN_VOICE_NAME}' not found")
    vid = cand[0]["voice_id"]
    print(f"[TTS] Resolved Eleven voice name '{ELEVEN_VOICE_NAME}' -> id={vid}")
    return vid

def _tts_elevenlabs(text: str) -> bytes:
    if not ELEVEN_API_KEY:
        raise RuntimeError("ELEVEN_API_KEY not set")
    voice_id = _eleven_resolve_voice_id()
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "model_id": ELEVEN_MODEL_ID,
        "voice_settings": {
            "stability": float(os.getenv("ELEVEN_STABILITY", "0.40")),
            "similarity_boost": float(os.getenv("ELEVEN_SIMILARITY", "0.85")),
            "style": float(os.getenv("ELEVEN_STYLE", "0.25")),
            "use_speaker_boost": os.getenv("ELEVEN_SPEAKER_BOOST", "true").lower() == "true",
        },
    }
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    # If it failed (e.g., you don’t have rights to that voice), they return JSON error
    if r.headers.get("content-type","").startswith("application/json") or r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"error": r.text}
        raise RuntimeError(f"ElevenLabs error: {err}")
    return r.content

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

def create_tts_audio(text: str) -> bytes:
    try:
        clean = _normalize_text(text)[:1200]
        sentences = _split_sentences(clean) or [clean or ""]
        plain = ". ".join(sentences)

        provider = os.getenv("TTS_PROVIDER", "").lower()

        if provider == "piper":
            return _tts_piper(plain)
        elif provider == "coqui":
            return _tts_coqui(plain)
        else:
            # Edge TTS fallback if you keep it around
            ssml = _to_ssml(sentences)
            mp3_raw = asyncio.run(_synth_ssml(ssml, VOICE_NAME))

            if not mp3_raw:
                return b""

            try:
                if TTS_POSTPROCESS:
                    return _polish_audio(mp3_raw)
                else:
                    return mp3_raw
            except Exception as e:
                print("[TTS] Post-process failed; returning raw audio:", e)
                return mp3_raw

    except Exception as e:
        print("[TTS] Error in create_tts_audio:", e)
        return b""



def send_to_discord(channel_id: str, text: str) -> None:
    """Send plain text message to Discord text channel."""
    url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages"
    headers = {"Authorization": f"Bot {DISCORD_TOKEN}"}
    payload = {"content": text}
    requests.post(url, headers=headers, json=payload, timeout=10)




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
    image_data = data.get("image")
    image_b64 = None
    if image_data:
        image_b64 = image_data.split(",", 1)[1] if "," in image_data else image_data

    history_text = build_history_for_guild(guild_id)
    prompt = (
        "Dette er en pågående samtale i en Discord-server.\n"
        + (f"Tidligere meldinger (kort):\n{history_text}\n\n" if history_text else "")
        + f"Nå sier {user_name}: {user_message}\n"
        "Svar naturlig på norsk og hold tråden i samtalen."
    )

    start = time.time()
    reply_raw = get_ai_response(
        prompt,
        user_id=user_id,
        user_name=user_name,
        guild_id=guild_id,
        image=image_b64,
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
    default_channel = DISCORD_TEXT_CHANNEL if DISCORD_TEXT_CHANNEL != "0" else None
    channel_id = request.form.get("channel_id") or default_channel
    user_name = request.form.get("user_name", "FAEEEEEN")
    user_id = (request.form.get("user_id") or "").strip() or None
    guild_id   = (request.form.get("guild_id") or "").strip() or None
    audio_file = request.files.get("file")
    image_data = request.form.get("image")
    image_b64 = None
    if image_data:
        image_b64 = image_data.split(",", 1)[1] if "," in image_data else image_data
    start_total = time.time()
    if not audio_file:
        return {"error": "no file"}, 400

    if not speech_recognition_enabled:
        return {"status": "ignored"}

    filename = audio_file.filename or "temp_audio"
    ext = os.path.splitext(filename)[1]
    path = f"temp_audio{ext}"
    audio_file.save(path)

    user_message = ""  # avoid UnboundLocalError in finally

    try:
    # Load and normalize to mono 16 kHz for overlap handling
        audio_seg = AudioSegment.from_file(path)
        audio_seg = audio_seg.set_channels(1).set_frame_rate(16000)

    # Prepend last tail for this user to avoid mid-word cuts
        if user_id:
            prev_tail = LAST_TAIL.get(user_id)
            if prev_tail:
                audio_seg = prev_tail + audio_seg
        # Store new tail from the end of this chunk
            LAST_TAIL[user_id] = audio_seg[-TAIL_MS:] if len(audio_seg) > TAIL_MS else audio_seg

    # Export to a real WAV path (don’t mix headers/extensions)
        wav_path = os.path.splitext(path)[0] + "_merged.wav"
        audio_seg.export(wav_path, format="wav")

        start = time.time()
        user_message = transcribe_audio(wav_path)
        last_process_times["speech_ms"] = int((time.time() - start) * 1000)

    except ValueError as err:
        return {"error": str(err)}, 400

    finally:
    # Clean up both temp files if present
        try:
            os.remove(path)
        except Exception:
            pass
        try:
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

        if not user_message.strip():
            return {"status": "ignored"}


    history_text = build_history_for_guild(guild_id)
    prompt = (
    "Dette er en pågående samtale i en Discord-server.\n"
    + (f"Tidligere meldinger (kort):\n{history_text}\n\n" if history_text else "")
    + f"Nå sier {user_name}: {user_message}\n"
    "Svar naturlig på norsk og hold tråden i samtalen."
)

    start = time.time()
    reply_raw = get_ai_response(
    prompt,
    user_id=user_id,
    user_name=user_name,
    guild_id=guild_id,
    image=image_b64,
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
