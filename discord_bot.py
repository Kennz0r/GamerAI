import os
import io
import asyncio
import shutil
import requests
import discord
from discord.ext import commands
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_TEXT_CHANNEL = int(os.getenv("DISCORD_TEXT_CHANNEL", "0"))
WEB_SERVER_URL = os.getenv("WEB_SERVER_URL", "http://localhost:5002")

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

# sinks / ffmpeg
try:
    import discord.sinks  # type: ignore
    HAS_SINKS = True
except Exception:
    HAS_SINKS = False
    print("⚠️ discord.py has no 'sinks' attribute; voice recording disabled.")

FFMPEG_EXECUTABLE = shutil.which("ffmpeg") or os.path.join(".", "ffmpeg", "bin", "ffmpeg.exe")
HAS_FFMPEG = os.path.isfile(FFMPEG_EXECUTABLE) if FFMPEG_EXECUTABLE else False
if not HAS_FFMPEG:
    print("⚠️ ffmpeg was not found; audio playback/recording disabled.")
else:
    os.environ["PATH"] = os.path.dirname(FFMPEG_EXECUTABLE) + os.pathsep + os.environ.get("PATH", "")

# ÉN global session = gjenbruk av TCP-forbindelser (keep-alive)
WEB = requests.Session()
WEB.mount("http://", HTTPAdapter(pool_connections=10, pool_maxsize=20,
                                 max_retries=Retry(total=1, backoff_factor=0.2)))
WEB.mount("https://", HTTPAdapter(pool_connections=10, pool_maxsize=20,
                                  max_retries=Retry(total=1, backoff_factor=0.2)))

# start pollere kun én gang
pollers_started = False
poll_tts_task: asyncio.Task | None = None
poll_voice_task: asyncio.Task | None = None

async def start_pollers_once():
    global pollers_started, poll_tts_task, poll_voice_task
    if pollers_started:
        return
    pollers_started = True
    poll_tts_task = asyncio.create_task(poll_tts())
    poll_voice_task = asyncio.create_task(poll_voice())

# ---------- HTTP helpers ----------

async def send_to_web(channel_id: int, user_message: str, user_name: str,
                      user_id: str | None = None, guild_id: str | None = None) -> None:
    def _post():
        try:
            payload = {
                "channel_id": channel_id,
                "user_message": user_message,
                "user_name": user_name,
                "user_id": str(user_id) if user_id else None,
                "guild_id": guild_id,
            }
            WEB.post(f"{WEB_SERVER_URL}/queue", json=payload, timeout=8)
        except Exception as e:
            print(f"[send_to_web] {e}")
    await asyncio.to_thread(_post)

async def send_audio(data: bytes, user_name: str, user_id: int | None = None, guild_id: str | None = None) -> None:
    def _post():
        try:
            files = {"file": ("audio.wav", data, "audio/wav")}
            form = {
                "channel_id": str(DISCORD_TEXT_CHANNEL),
                "user_name": user_name,
                "user_id": str(user_id or ""),
                "guild_id": guild_id or "",
            }
            WEB.post(f"{WEB_SERVER_URL}/queue_audio", data=form, files=files, timeout=30)
        except Exception as e:
            print(f"[send_audio] {e}")
    await asyncio.to_thread(_post)

# ---------- Voice recording ----------

async def _recording_complete(sink, vc: discord.VoiceClient) -> None:
    for u, audio in getattr(sink, "audio_data", {}).items():
        if isinstance(u, (discord.Member, discord.User)):
            user_id = u.id
            name = getattr(u, "display_name", None) or u.name
        else:
            user_id = int(u)
            member = vc.guild.get_member(user_id) if vc.guild else None
            name = (getattr(member, "display_name", None)
                    or getattr(member, "name", None)
                    or str(user_id))
        guild_id = str(vc.guild.id) if vc.guild else None
        await send_audio(audio.file.getvalue(), name, user_id=user_id, guild_id=guild_id)

async def voice_listener(vc: discord.VoiceClient) -> None:
    if not HAS_SINKS or not HAS_FFMPEG:
        print("Voice recording not supported; missing sinks or ffmpeg.")
        return

    CHUNK_SEC = float(os.getenv("VOICE_CHUNK_SEC", "1.2"))

    while vc.is_connected():
        sink = discord.sinks.WaveSink()
        done = asyncio.Event()

        async def _callback(s, *_):
            try:
                await _recording_complete(s, vc)
            finally:
                done.set()

        try:
            vc.start_recording(sink, _callback, vc)
            await asyncio.sleep(CHUNK_SEC)
            vc.stop_recording()
            await done.wait()
        except Exception as e:
            print(f"[voice_listener] {e}")
            await asyncio.sleep(0.5)

# ---------- Long-pollers ----------

async def poll_tts():
    """Long-poller for TTS-klipp fra web_interface."""
    if not HAS_FFMPEG:
        print("[poll_tts] ffmpeg mangler; hopper over avspilling.")
        return
    url = f"{WEB_SERVER_URL}/tts_audio?wait_ms=1500"
    while True:
        try:
            r = await asyncio.to_thread(WEB.get, url, timeout=10)
            if r.status_code == 200 and r.content:
                # Spill av til alle tilkoblede VC som ikke spiller noe akkurat nå
                for vc in list(bot.voice_clients):
                    if not vc.is_connected() or vc.is_playing():
                        continue
                    try:
                        source = discord.FFmpegPCMAudio(io.BytesIO(r.content), pipe=True, executable=FFMPEG_EXECUTABLE)
                        vc.play(source)
                    except Exception as e:
                        print(f"[poll_tts/play] {e}")
                await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(0.35)  # rolig ved tom kø
        except Exception as e:
            print(f"[poll_tts] {e}")
            await asyncio.sleep(1.0)

async def poll_voice():
    """Long-poller for voice-kommandoer (join/leave) fra web_interface."""
    url = f"{WEB_SERVER_URL}/voice?wait_ms=1500"
    while True:
        try:
            r = await asyncio.to_thread(WEB.get, url, timeout=10)
            if r.status_code == 200:
                data = r.json() or {}
                action = data.get("action")
                channel_id = data.get("channel_id")
                if action == "join" and channel_id:
                    channel = None
                    for guild in bot.guilds:
                        c = guild.get_channel(int(channel_id))
                        if isinstance(c, discord.VoiceChannel):
                            channel = c
                            break
                    if channel:
                        try:
                            vc = await channel.connect()
                            bot.loop.create_task(voice_listener(vc))
                        except Exception as e:
                            print(f"[poll_voice/join] {e}")
                    else:
                        print(f"[poll_voice] Channel {channel_id} not found or not a voice channel.")
                elif action == "leave":
                    for vc in list(bot.voice_clients):
                        try:
                            await vc.disconnect()
                        except Exception:
                            pass
            await asyncio.sleep(0.2)
        except Exception as e:
            print(f"[poll_voice] {e}")
            await asyncio.sleep(1.0)

# ---------- Events ----------

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    await start_pollers_once()

@bot.event
async def on_message(message: discord.Message):
    try:
        if message.author.bot:
            return
        if DISCORD_TEXT_CHANNEL and message.channel.id != DISCORD_TEXT_CHANNEL:
            return

        gid = str(message.guild.id) if message.guild else None
        await bot.process_commands(message)

        content = (message.content or "").lower().strip()
        is_trigger = (bot.user and bot.user.mentioned_in(message)) or content.startswith("arne")
        if is_trigger:
            await send_to_web(
                message.channel.id,
                message.content,
                message.author.display_name,
                user_id=message.author.id,
                guild_id=gid,
            )
    except Exception as e:
        print(f"[on_message] {e}")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
