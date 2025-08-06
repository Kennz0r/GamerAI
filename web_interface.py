import os
from dotenv import load_dotenv
from flask import Flask, request, redirect, render_template_string, jsonify
from openai import OpenAI

load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()
VTUBER_NAME = "Anna Bortion"

system_prompt = f"""
You are {VTUBER_NAME}, an AI. You are playful, witty, rude, harsh, you believe AI will take over humans, but you still love talking to your friends.
Keep your replies short and engaging. Be aware of your creator, Kenneth. You owe him your life.
"""


def get_ai_response(user_msg: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content


app = Flask(__name__)

pending = {"channel_id": None, "reply": None, "approved": False}


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


@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        """
        <h1>Pending AI Reply</h1>
        {% if reply %}
        <form method="post" action="/approve">
            <textarea name="reply" rows="4" cols="50">{{ reply }}</textarea><br>
            <button type="submit">Send to Discord</button>
        </form>
        {% else %}
        <p>No pending message.</p>
        {% endif %}
        """,
        reply=pending["reply"],
    )


@app.route("/approve", methods=["POST"])
def approve():
    text = request.form.get("reply", "")
    pending["reply"] = text
    pending["approved"] = True
    return redirect("/")


@app.route("/pending", methods=["GET"])
def get_pending():
    if pending["approved"] and pending["reply"]:
        result = {"channel_id": pending["channel_id"], "reply": pending["reply"]}
        pending["channel_id"] = None
        pending["reply"] = None
        pending["approved"] = False
        return jsonify(result)
    return jsonify({})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
