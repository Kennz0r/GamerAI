import os
import io
import asyncio
import shutil
import requests
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
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)
try:
    import discord.sinks  # type: ignore
    HAS_SINKS = True
except Exception:
    HAS_SINKS = False
    print("⚠️ discord.py has no 'sinks' attribute; voice recording disabled.")

FFMPEG_EXECUTABLE = shutil.which("ffmpeg") or os.path.join(".", "ffmpeg", "bin", "ffmpeg.exe")
HAS_FFMPEG = os.path.isfile(FFMPEG_EXECUTABLE) if FFMPEG_EXECUTABLE else False
if not HAS_FFMPEG:
    print("⚠️ ffmpeg was not found; audio recording disabled.")
else:
    # Ensure the directory containing ffmpeg is on PATH so discord.sinks can find it
    ffmpeg_dir = os.path.dirname(FFMPEG_EXECUTABLE)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")



async def send_to_web(channel_id: int, user_message: str, user_name: str,
                      user_id: str | None = None, guild_id: str | None = None) -> None:
    def _post():
        try:
            requests.post(
                f"{WEB_SERVER_URL}/queue",
                json={
                    "channel_id": channel_id,
                    "user_message": user_message,
                    "user_name": user_name,
                    "user_id": user_id,
                    "guild_id": guild_id,  # <--
                },
                timeout=5,
            )
        except Exception as e:
            print(f"Error sending to web server: {e}")
    await asyncio.to_thread(_post)



async def send_audio(data: bytes, user_name: str,
                     user_id: int | None = None, guild_id: str | None = None) -> None:
    def _post():
        try:
            files = {"file": ("audio.mp3", data, "audio/mpeg")}
            requests.post(
                f"{WEB_SERVER_URL}/queue_audio",
                data={
                    "channel_id": DISCORD_TEXT_CHANNEL,
                    "user_name": user_name,
                    "user_id": str(user_id or ""),
                    "guild_id": guild_id or "",  # <--
                },
                files=files,
                timeout=360,
            )
        except Exception as e:
            print(f"Error sending audio: {e}")
    await asyncio.to_thread(_post)



async def _recording_complete(sink, vc: discord.VoiceClient) -> None:
    for u, audio in getattr(sink, "audio_data", {}).items():
        # resolve name & user_id like we did previously...
        if isinstance(u, (discord.Member, discord.User)):
            user_id = u.id
            name = getattr(u, "display_name", None) or u.name
        else:
            user_id = int(u)
            member = vc.guild.get_member(user_id) if vc.guild else None
            name = (getattr(member, "display_name", None)
                    or getattr(member, "name", None)
                    or str(user_id))

        guild_id = str(vc.guild.id) if vc.guild else None  # <--
        await send_audio(audio.file.getvalue(), name, user_id=user_id, guild_id=guild_id)





async def voice_listener(vc: discord.VoiceClient) -> None:
    """Continuously record short audio snippets and send them for processing."""
    if not HAS_SINKS or not HAS_FFMPEG:
        print("Voice recording not supported; missing sinks or ffmpeg.")
        return

    while vc.is_connected():
        sink = discord.sinks.MP3Sink()

        done = asyncio.Event()

        async def _callback(s, *_):
            await _recording_complete(s, vc)
            done.set()

        try:
            vc.start_recording(sink, _callback, vc)
            await asyncio.sleep(5)  # Record 5 seconds
            vc.stop_recording()
            await done.wait()  # Wait until _callback finishes before continuing
        except Exception as e:
            print(f"Recording error: {e}")
            break


@tasks.loop(seconds=5)
async def poll_tts():
    def _get():
        try:
            r = requests.get(f"{WEB_SERVER_URL}/tts_audio", timeout=5)
            if r.status_code == 200:
                return r.content
        except Exception as e:
            print(f"Error fetching TTS audio: {e}")
        return None

    data = await asyncio.to_thread(_get)
    if not data:
        return

    for vc in list(bot.voice_clients):
        if not vc.is_connected() or vc.is_playing():
            continue
        try:
            source = discord.FFmpegPCMAudio(io.BytesIO(data), pipe=True, executable=FFMPEG_EXECUTABLE)
            vc.play(source)
        except Exception as e:
            print(f"Error playing TTS: {e}")




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
    #print(f"poll_voice received: {data}")

    action = data.get("action")
    channel_id = data.get("channel_id")

    if action == "join" and channel_id:
        print(f"Trying to join voice channel {channel_id}")
        channel = None
        for guild in bot.guilds:
            c = guild.get_channel(int(channel_id))
            if isinstance(c, discord.VoiceChannel):
                channel = c
                break

        if channel:
            print(f"Found voice channel: {channel.name}")
            vc = await channel.connect()
            print(f"Connected to: {channel.name}")
            bot.loop.create_task(voice_listener(vc))
        else:
            print(f"Channel ID {channel_id} is not a voice channel or not found.")

    elif action == "leave":
        for vc in list(bot.voice_clients):
            await vc.disconnect()
            print("Disconnected from voice channel.")




@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    poll_voice.start()
    poll_tts.start()


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if DISCORD_TEXT_CHANNEL and message.channel.id != DISCORD_TEXT_CHANNEL:
        return
    await bot.process_commands(message)

    content = message.content.lower()
    if bot.user.mentioned_in(message) or content.startswith("anna"):
        await send_to_web(
            message.channel.id, message.content, message.author.display_name, user_id=message.author.id, guild_id=str(message.guild.id) if message.guild else None,  # <--
        )


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
