import os
from dotenv import load_dotenv
from flask import Flask, request, redirect, render_template_string, jsonify

from ai import get_ai_response, transcribe_audio

load_dotenv()


app = Flask(__name__)

pending = {"channel_id": None, "reply": None, "approved": False}
voice_command = {"action": None, "channel_id": None}


@app.route("/queue", methods=["POST"])
def queue_message():
    data = request.get_json(force=True)
    channel_id = data.get("channel_id")
    user_message = data.get("user_message", "")
    reply = get_ai_response(user_message)
    pending["channel_id"] = channel_id
    pending["reply"] = reply
    pending["approved"] = False
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
    pending["channel_id"] = channel_id
    pending["reply"] = reply
    pending["approved"] = False
    return {"status": "queued"}


@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        """
        <h1>Pending AI Reply</h1>
        {% if reply %}
        <form method="post" action="approve">
            <textarea name="reply" rows="4" cols="50">{{ reply }}</textarea><br>
            <button type="submit">Send to Discord</button>
        </form>
        {% else %}
        <p>No pending message.</p>
        {% endif %}
        <h2>Voice Control</h2>
        <form method="post" action="voice">
            <input type="hidden" name="action" value="join" />
            <input type="text" name="channel_id" placeholder="Voice Channel ID" /><br>
            <button type="submit">Join Voice</button>
        </form>
        <form method="post" action="voice">
            <input type="hidden" name="action" value="leave" />
            <button type="submit">Leave Voice</button>
        </form>
        """,
        reply=pending["reply"],
    )


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
