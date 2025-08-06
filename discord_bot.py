import os
import asyncio
import requests
from gtts import gTTS
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_TEXT_CHANNEL = int(os.getenv("DISCORD_TEXT_CHANNEL", "0"))
WEB_SERVER_URL = os.getenv("WEB_SERVER_URL", "http://localhost:5002")

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
bot = commands.Bot(command_prefix="!", intents=intents)
try:
    import discord.sinks  # type: ignore
    HAS_SINKS = True
except Exception:
    HAS_SINKS = False
    print("⚠️ discord.py has no 'sinks' attribute; voice recording disabled.")


def create_tts_file(text, filename="response.mp3"):
    try:
        print(f"🗣 Creating TTS for: {text}")
        tts = gTTS(text=text, lang="no")
        tts.save(filename)
        print(f"✅ TTS saved as {os.path.abspath(filename)}")
    except Exception as e:
        print(f"❌ Failed to create TTS: {e}")
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

    print(f"🗣 Creating TTS for: {reply}")
    path = create_tts_file(reply)  # <-- This should stay synchronous
    print(f"✅ TTS saved as {os.path.abspath(path)}")

    channel = bot.get_channel(int(channel_id))
    if not channel:
        print("❌ Channel not found")
        return

    try:
        await channel.send(reply)

        vc = channel.guild.voice_client
        if vc and not vc.is_playing():
            FFMPEG_PATH = "./ffmpeg/bin/ffmpeg.exe"
            print(f"🎧 Playing audio from: {path}")
            source = discord.FFmpegPCMAudio(path, executable=FFMPEG_PATH)
            vc.play(source)
        else:
            print("📎 Sending MP3 as file (no voice client)")
            await channel.send(file=discord.File(path))

        # Optional: wait for FFmpeg to finish before deleting
        await asyncio.sleep(3)

    except Exception as e:
        print(f"⚠️ Error playing or sending TTS: {e}")
    finally:
        if os.path.exists(path):
            os.remove(path)
            print("🧹 Cleaned up mp3")
        else:
            print("❌ File missing before cleanup")


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


def _recording_complete(sink, vc: discord.VoiceClient) -> None:
    """Callback when a recording chunk is finished."""
    for audio in getattr(sink, "audio_data", {}).values():
        bot.loop.create_task(send_audio(audio.file.getvalue()))


async def voice_listener(vc: discord.VoiceClient) -> None:
    """Continuously record audio from a voice client and send it for processing."""
    if not HAS_SINKS:
        print("Voice recording not supported in this discord.py version.")
        return

    while vc.is_connected():
        sink = discord.sinks.MP3Sink()
        vc.start_recording(sink, _recording_complete, vc)
        await asyncio.sleep(5)
        vc.stop_recording()



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
    print(f"📡 poll_voice received: {data}")

    action = data.get("action")
    channel_id = data.get("channel_id")

    if action == "join" and channel_id:
        print(f"➡️ Trying to join voice channel {channel_id}")
        channel = None
        for guild in bot.guilds:
            c = guild.get_channel(int(channel_id))
            if isinstance(c, discord.VoiceChannel):
                channel = c
                break

        if channel:
            print(f"✅ Found voice channel: {channel.name}")
            vc = await channel.connect()
            print(f"🎤 Connected to: {channel.name}")
            bot.loop.create_task(voice_listener(vc))
        else:
            print(f"⚠️ Channel ID {channel_id} is not a voice channel or not found.")

    elif action == "leave":
        for vc in list(bot.voice_clients):
            await vc.disconnect()
            print("👋 Disconnected from voice channel.")




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
