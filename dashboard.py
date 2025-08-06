from flask import Flask, request, redirect, render_template_string
import asyncio
import os
import discord

pending = {"reply": None, "channel_id": None}

def run_dashboard(bot, create_tts_file):
    app = Flask(__name__)

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
        if pending["channel_id"]:
            asyncio.run_coroutine_threadsafe(send_pending_to_discord(), bot.loop)
        return redirect("/")

    async def send_pending_to_discord():
        channel = bot.get_channel(pending["channel_id"])
        reply = pending["reply"]
        path = create_tts_file(reply)
        try:
            await channel.send(reply)
            if channel.guild.voice_client:
                source = discord.FFmpegPCMAudio(path)
                channel.guild.voice_client.play(source)
            else:
                await channel.send(file=discord.File(path))
        finally:
            os.remove(path)
            pending["reply"] = None
            pending["channel_id"] = None

    app.run(host="0.0.0.0", port=5000)
