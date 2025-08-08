import os
import io
import subprocess
import sys, inspect
import signal
import json
import requests
from dotenv import load_dotenv
from flask import Flask, request, redirect, jsonify, send_from_directory
from ai import get_ai_response, transcribe_audio, set_model, ollama_client

import asyncio, re
import regex as reg
from unidecode import unidecode
from pydub import AudioSegment, effects
import io
import edge_tts
from flask import Response

load_dotenv()

DISCORD_TEXT_CHANNEL = os.getenv("DISCORD_TEXT_CHANNEL", "0")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_API_BASE = "https://discord.com/api/v10"

VOICE_NAME = os.getenv("TTS_VOICE", "nb-NO-IselinNeural")  # or nb-NO-FinnNeural / en-US-AnaNeural
TTS_RATE  = os.getenv("TTS_RATE", "0%")   # slightly slower, more natural
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz") # neutral pitch

URL_RE = re.compile(r"https?://\S+")
CODE_FENCE = re.compile(r"```.*?```", re.S)
EMOJI_RE = reg.compile(r"\p{Emoji_Presentation}")

# --- ElevenLabs config ---
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs").lower()
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "").strip()
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "").strip()
ELEVEN_VOICE_NAME = os.getenv("ELEVEN_VOICE_NAME", "").strip()
ELEVEN_MODEL_ID = os.getenv("ELEVEN_MODEL_ID", "eleven_multilingual_v2")

app = Flask(__name__)

pending = {"channel_id": None, "reply": None}
voice_command = {"action": None, "channel_id": None}
conversation = []
# Track whether speech recognition is enabled
speech_recognition_enabled = True
# Pending TTS audio bytes for Discord bot and web preview
pending_tts_discord: bytes | None = None
# Pending preview audio is generated on-demand but keep storage for compatibility
pending_tts_web: bytes | None = None
# Handle for the optional Discord bot subprocess
discord_bot_process: subprocess.Popen | None = None
# Storage for optional fine-tuning examples

try:
    print("[TTS] Python:", sys.executable)
    import edge_tts
    print("[TTS] edge-tts file:", edge_tts.__file__)
    from inspect import signature
    print("[TTS] edge-tts Communicate params:", list(signature(edge_tts.Communicate).parameters.keys()))
except Exception as e:
    print("[TTS] edge-tts introspection failed:", e)

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
        # ElevenLabs handles long strings fine; chunking still helps cadence
        sentences = _split_sentences(clean) or [clean or ""]
        plain = " ".join(sentences)

        if TTS_PROVIDER == "elevenlabs":
            mp3_raw = _tts_elevenlabs(plain)
        else:
            # fallback to your existing Edge path if you keep it
            ssml = _to_ssml(sentences)
            mp3_raw = asyncio.run(_synth_ssml(ssml, VOICE_NAME))

        if not mp3_raw:
            return b""
        try:
            return _polish_audio(mp3_raw)
        except Exception as e:
            print("[TTS] Post-process failed; returning raw audio:", e)
            return mp3_raw
    except Exception as e:
        print("[TTS] TTS error:", e)
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
    channel_id = data.get("channel_id")
    user_message = data.get("user_message", "")
    user_name = data.get("user_name", "Unknown")
    reply = get_ai_response(f"{user_name} sier: {user_message}")
    conversation.append({
        "user_name": user_name,
        "user_message": user_message,
        "reply": reply,
        "rating": None,
    })
    pending["channel_id"] = channel_id
    pending["reply"] = reply

    # Generate TTS preview automatically so the user can hear the reply
    global pending_tts_web, pending_tts_discord
    pending_tts_web = create_tts_audio(reply)
    # Clear any pending Discord audio until the message is approved
    pending_tts_discord = None

    return {"status": "queued"}


@app.route("/queue_audio", methods=["POST"])
def queue_audio():
    channel_id = request.form.get("channel_id") or None
    user_name = request.form.get("user_name", "Voice")
    audio_file = request.files.get("file")
    if not audio_file:
        return {"error": "no file"}, 400

    if not speech_recognition_enabled:
        return {"status": "ignored"}

    filename = audio_file.filename or "temp_audio"
    ext = os.path.splitext(filename)[1]
    path = f"temp_audio{ext}"
    audio_file.save(path)

    try:
        user_message = transcribe_audio(path)
    except ValueError as err:
        return {"error": str(err)}, 400
    finally:
        os.remove(path)

    
    if not user_message.strip():
        return {"status": "ignored"}


    reply = get_ai_response(f"{user_name} sier: {user_message}")
    conversation.append({
        "user_name": user_name,
        "user_message": user_message,
        "reply": reply,
        "rating": None,
    })
    pending["channel_id"] = channel_id
    pending["reply"] = reply

    # Prepare TTS preview for voice input replies as well
    global pending_tts_web, pending_tts_discord
    pending_tts_web = create_tts_audio(reply)
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

    if channel_id and DISCORD_TOKEN:
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


@app.route("/discord_bot", methods=["POST"])
def control_discord_bot():
    global discord_bot_process
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
