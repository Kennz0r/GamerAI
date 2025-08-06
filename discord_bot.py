import os
import asyncio
import requests
from gtts import gTTS
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

try:
    # `discord.sinks` is only available in certain discord.py forks and versions.
    # Attempt to import it but fall back gracefully if unavailable so the bot
    # can still run without voice recording features.
    from discord import sinks  # type: ignore
except Exception:  # pragma: no cover - handles ImportError and AttributeError
    sinks = None  # type: ignore

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_TEXT_CHANNEL = int(os.getenv("DISCORD_TEXT_CHANNEL", "0"))
WEB_SERVER_URL = os.getenv("WEB_SERVER_URL", "http://localhost:5000")

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
bot = commands.Bot(command_prefix="!", intents=intents)


def create_tts_file(text, filename="response.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename


async def send_to_web(channel_id: int, user_message: str) -> None:
    def _post():
        try:
            requests.post(
                f"{WEB_SERVER_URL}/queue",
                json={"channel_id": channel_id, "user_message": user_message},
                timeout=5,
            )
        except Exception as e:
            print(f"Error sending to web server: {e}")
    await asyncio.to_thread(_post)


@tasks.loop(seconds=5)
async def fetch_pending():
    def _get():
        try:
            r = requests.get(f"{WEB_SERVER_URL}/pending", timeout=5)
            return r.json()
        except Exception as e:
            print(f"Error fetching from web server: {e}")
            return {}
    data = await asyncio.to_thread(_get)
    reply = data.get("reply")
    channel_id = data.get("channel_id")
    if not reply or not channel_id:
        return
    channel = bot.get_channel(int(channel_id))
    if not channel:
        return
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


async def send_audio(data: bytes) -> None:
    def _post():
        try:
            files = {"file": ("audio.mp3", data, "audio/mpeg")}
            requests.post(
                f"{WEB_SERVER_URL}/queue_audio",
                data={"channel_id": DISCORD_TEXT_CHANNEL},
                files=files,
                timeout=10,
            )
        except Exception as e:
            print(f"Error sending audio: {e}")

    await asyncio.to_thread(_post)


async def voice_loop(vc: discord.VoiceClient):
    """Continuously record audio from a voice channel and forward it to the web server.

    If the installed discord.py version does not support voice sinks the loop exits
    immediately.
    """

    if not sinks:  # Voice recording is unsupported
        return

    while vc.is_connected():
        sink = sinks.MP3Sink()
        vc.start_recording(sink, lambda *args: None)
        await asyncio.sleep(5)
        vc.stop_recording()
        audio = next(iter(sink.audio_data.values()), None)
        if audio:
            await send_audio(audio.file.getvalue())


@tasks.loop(seconds=5)
async def poll_voice():
    def _get():
        try:
            r = requests.get(f"{WEB_SERVER_URL}/voice", timeout=5)
            return r.json()
        except Exception as e:
            print(f"Error fetching voice command: {e}")
            return {}

    data = await asyncio.to_thread(_get)
    action = data.get("action")
    channel_id = data.get("channel_id")

    if action == "join" and channel_id and sinks:
        channel = bot.get_channel(int(channel_id))
        if isinstance(channel, discord.VoiceChannel):
            vc = await channel.connect()
            bot.loop.create_task(voice_loop(vc))
    elif action == "leave":
        for vc in list(bot.voice_clients):
            await vc.disconnect()


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    fetch_pending.start()
    poll_voice.start()


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if DISCORD_TEXT_CHANNEL and message.channel.id != DISCORD_TEXT_CHANNEL:
        return
    await send_to_web(message.channel.id, message.content)
    await bot.process_commands(message)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
