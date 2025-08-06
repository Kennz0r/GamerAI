import os
from dotenv import load_dotenv
from flask import Flask, request, redirect, jsonify, send_from_directory

from ai import get_ai_response, transcribe_audio

load_dotenv()


app = Flask(__name__)

pending = {"channel_id": None, "reply": None, "approved": False}
voice_command = {"action": None, "channel_id": None}
conversation = []


@app.route("/queue", methods=["POST"])
def queue_message():
    data = request.get_json(force=True)
    channel_id = data.get("channel_id")
    user_message = data.get("user_message", "")
    reply = get_ai_response(user_message)
    conversation.append({"user": user_message, "reply": reply})
    pending["channel_id"] = channel_id
    pending["reply"] = reply
    pending["approved"] = True
    return {"status": "queued"}


@app.route("/queue_audio", methods=["POST"])
def queue_audio():
    channel_id = request.form.get("channel_id")
    audio_file = request.files.get("file")
    if not audio_file:
        return {"error": "no file"}, 400
    path = "temp_audio"
    audio_file.save(path)
    try:
        user_message = transcribe_audio(path)
    finally:
        os.remove(path)
    reply = get_ai_response(user_message)
    conversation.append({"user": user_message, "reply": reply})
    pending["channel_id"] = channel_id
    pending["reply"] = reply
    pending["approved"] = True
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


@app.route("/approve", methods=["POST"])
def approve():
    text = request.form.get("reply", "")
    pending["reply"] = text
    pending["approved"] = True
    return redirect(".")


@app.route("/pending", methods=["GET"])
def get_pending():
    if pending["approved"] and pending["reply"]:
        result = {"channel_id": pending["channel_id"], "reply": pending["reply"]}
        pending["channel_id"] = None
        pending["reply"] = None
        pending["approved"] = False
        return jsonify(result)
    return jsonify({})


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
