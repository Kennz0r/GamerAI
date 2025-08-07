import os
import io
import subprocess
import sys
import requests
from dotenv import load_dotenv
from flask import Flask, request, redirect, jsonify, send_from_directory
from gtts import gTTS

from ai import get_ai_response, transcribe_audio

load_dotenv()

DISCORD_TEXT_CHANNEL = os.getenv("DISCORD_TEXT_CHANNEL", "0")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_API_BASE = "https://discord.com/api/v10"

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


def create_tts_audio(text: str) -> bytes:
    tts = gTTS(text=text, lang="no")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


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
    conversation.append({"user_name": user_name, "user_message": user_message, "reply": reply})
    pending["channel_id"] = channel_id
    pending["reply"] = reply
    return {"status": "queued"}


@app.route("/queue_audio", methods=["POST"])
def queue_audio():
    channel_id = request.form.get("channel_id") or DISCORD_TEXT_CHANNEL
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

    reply = get_ai_response(f"{user_name} sier: {user_message}")
    conversation.append({"user_name": user_name, "user_message": user_message, "reply": reply})
    pending["channel_id"] = channel_id
    pending["reply"] = reply
    return {"status": "queued"}


@app.route("/", methods=["GET"])
def index():
    return send_from_directory("static", "index.html")


@app.route("/conversation", methods=["GET"])
def get_conversation():
    return jsonify(conversation)


@app.route("/pending_message", methods=["GET"])
def get_pending_message():
    return jsonify({"reply": pending["reply"]})


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
        return data, 200, {"Content-Type": "audio/mpeg"}
    return ("", 204)


@app.route("/tts_preview", methods=["GET", "POST"])
def get_tts_preview():
    global pending_tts_web
    if request.method == "POST":
        data = request.get_json(force=True)
        text = data.get("text", "")
        audio = create_tts_audio(text)
        return audio, 200, {"Content-Type": "audio/mpeg"}
    if pending_tts_web:
        data = pending_tts_web
        pending_tts_web = None
        return data, 200, {"Content-Type": "audio/mpeg"}
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
    channel_id = pending.get("channel_id") or DISCORD_TEXT_CHANNEL

    if conversation:
        conversation[-1]["reply"] = pending_reply

    # Generate TTS audio and store for playback
    global pending_tts_discord, pending_tts_web
    audio = create_tts_audio(pending_reply)
    pending_tts_discord = audio
    pending_tts_web = audio

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
            discord_bot_process.terminate()
            try:
                discord_bot_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                discord_bot_process.kill()
            finally:
                discord_bot_process = None
            return jsonify({"status": "stopped"})
        return jsonify({"status": "not_running"})
    return jsonify({"error": "unknown action"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
