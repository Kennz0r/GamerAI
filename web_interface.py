import os
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


def create_tts_file(text: str, filename: str = "response.mp3") -> str:
    tts = gTTS(text=text, lang="no")
    tts.save(filename)
    return filename


def send_to_discord(channel_id: str, text: str, path: str) -> None:
    url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages"
    headers = {"Authorization": f"Bot {DISCORD_TOKEN}"}
    with open(path, "rb") as f:
        requests.post(
            url,
            headers=headers,
            data={"content": text},
            files={"file": (os.path.basename(path), f, "audio/mpeg")},
            timeout=10,
        )
    os.remove(path)


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

    path = create_tts_file(pending_reply)
    send_to_discord(channel_id, pending_reply, path)

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
