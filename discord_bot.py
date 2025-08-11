import os
import io, wave
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
WEB_SERVER_URL = os.getenv("WEB_SERVER_URL", "http://127.0.0.1:5002")

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
def _mk_session():
    s = requests.Session()
    # pool_maxsize=1 ensures each long-poller keeps one reusable socket only
    s.mount("http://",  HTTPAdapter(pool_connections=1, pool_maxsize=1, pool_block=True,
                                    max_retries=Retry(total=1, backoff_factor=0.2)))
    s.mount("https://", HTTPAdapter(pool_connections=1, pool_maxsize=1, pool_block=True,
                                    max_retries=Retry(total=1, backoff_factor=0.2)))
    return s

WEB_POST  = _mk_session()  # queue + queue_audio
WEB_TTS   = _mk_session()  # /tts_audio long-poll
WEB_VOICE = _mk_session()  # /voice long-poll

# start pollere kun én gang
pollers_started = False
poll_tts_task: asyncio.Task | None = None
poll_voice_task: asyncio.Task | None = None

def _ensure_wav(pcm_or_wav: bytes, sr: int = 48000, ch: int = 2, sw: int = 2) -> bytes:
    # Hvis det allerede er en WAV: returner som er
    if len(pcm_or_wav) >= 12 and pcm_or_wav[:4] == b"RIFF" and pcm_or_wav[8:12] == b"WAVE":
        return pcm_or_wav
    # Ellers: pakk rå PCM inn i WAV-container
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(sw)   # 2 bytes = s16le
        wf.setframerate(sr)
        wf.writeframes(pcm_or_wav)
    return buf.getvalue()


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
            WEB_POST.post(f"{WEB_SERVER_URL}/queue", json=payload, timeout=8)
        except Exception as e:
            print(f"[send_to_web] {e}")
    await asyncio.to_thread(_post)

CURRENT_GUILD_ID: int | None = None

async def send_audio(data: bytes, user_name: str, user_id: int | None = None, guild_id: int | str | None = None) -> None:
    def _post():
        try:
            files = {"file": ("chunk.wav", data, "audio/wav")}
            form = {
                "channel_id": str(DISCORD_TEXT_CHANNEL),
                "user_name": user_name,
                "user_id": str(user_id or ""),
                "guild_id": str(guild_id or ""),
            }
            WEB_POST.post(f"{WEB_SERVER_URL}/queue_audio", data=form, files=files, timeout=30)
        except Exception as e:
            print(f"[send_audio] {e}")
    await asyncio.to_thread(_post)




def make_sink():
    # WAV (PCM) gir best STT. Bruk MP3 kun hvis du må spare båndbredde.
    return discord.sinks.WaveSink()  # type: ignore
async def voice_listener(vc: discord.VoiceClient):
    import discord.sinks  # type: ignore
    CHUNK_SEC = float(os.getenv("VOICE_CHUNK_SEC", "2.0"))

    while vc.is_connected():
        done = asyncio.Event()

        async def _callback(sink: "discord.sinks.Sink", *_):
            try:
                for _, ad in list(sink.audio_data.items()):
                    f = ad.file
                    try:
                        if hasattr(f, "flush"): f.flush()
                        if hasattr(f, "seek"):  f.seek(0)
                        data = f.read()
                    except Exception:
                        data = getattr(f, "getvalue", lambda: b"")()

                    if not data:
                        continue

                    # --- Tving korrekt WAV ---
                    def force_wav(b: bytes) -> bytes:
                        try:
                            # hvis det ser ut som WAV: finn 'data'-chunk og bruk bare rå PCM etter den
                            if len(b) > 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE":
                                i = b.find(b"data")
                                if i != -1 and i + 8 < len(b):
                                    raw = b[i+8:]
                                    if raw:
                                        return _ensure_wav(raw, sr=48000, ch=1, sw=2)
                            # ellers: pakk alt som rå PCM
                            return _ensure_wav(b, sr=48000, ch=1, sw=2)
                        except Exception:
                            return _ensure_wav(b, sr=48000, ch=1, sw=2)

                    wav_bytes = force_wav(data)

                    user = getattr(ad, "user", None)
                    uid = getattr(user, "id", None)
                    uname = getattr(user, "display_name", None) or getattr(user, "name", None) or "Voice"

                    await send_audio(
                        wav_bytes,
                        user_name=uname,
                        user_id=uid,
                        guild_id=vc.guild.id if vc.guild else None
                    )
            finally:
                try: sink.cleanup()
                except Exception: pass
                done.set()


        # start én chunk
        try:
            sink = discord.sinks.WaveSink()
            vc.start_recording(sink, _callback)
        except Exception as e:
            print(f"[voice_listener] start_recording failed: {e}")
            await asyncio.sleep(1.0)
            continue

        # la den ta opp litt
        await asyncio.sleep(CHUNK_SEC)

        # stopp (én gang), så venter vi på callback
        try:
            vc.stop_recording()
        except Exception as e:
            print(f"[voice_listener] stop_recording: {e}")

        try:
            await asyncio.wait_for(done.wait(), timeout=5)
        except asyncio.TimeoutError:
            print("[voice_listener] callback timeout")




# ---------- Long-pollers ----------
async def poll_tts():
    if not HAS_FFMPEG:
        print("[poll_tts] ffmpeg mangler; hopper over avspilling.")
        return
    while True:
        try:
            def _get():
                with WEB_TTS.get(f"{WEB_SERVER_URL}/tts_audio?wait_ms=5000", timeout=15) as r:
                    return r.content if r.status_code == 200 else None
            data = await asyncio.to_thread(_get)
            if data:
                for vc in list(bot.voice_clients):
                    if vc.is_connected() and not vc.is_playing():
                        try:
                            source = discord.FFmpegPCMAudio(io.BytesIO(data), pipe=True, executable=FFMPEG_EXECUTABLE)
                            vc.play(source)
                        except Exception as e:
                            print(f"[poll_tts/play] {e}")
                await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(0.4)
        except Exception as e:
            print(f"[poll_tts] {e}")
            await asyncio.sleep(1.0)

async def poll_voice():
    """Long-poller for voice-kommandoer (join/leave) fra web_interface."""
    while True:
        try:
            def _get():
                # 'with' sikrer at socket og respons lukkes
                with WEB_VOICE.get(f"{WEB_SERVER_URL}/voice?wait_ms=5000", timeout=15) as r:
                    if r.status_code == 200:
                        return r.json() or {}
                    return {}
            data = await asyncio.to_thread(_get)

            action = data.get("action")
            channel_id = data.get("channel_id")

            if action == "join" and channel_id:
                # Finn voice-kanalen
                target = None
                for guild in bot.guilds:
                    c = guild.get_channel(int(channel_id))
                    if isinstance(c, discord.VoiceChannel):
                        target = c
                        break
                if not target:
                    print(f"[poll_voice] Channel {channel_id} not found or not a voice channel.")
                else:
                    # Gjenbruk eksisterende VC om mulig (unngå ekstra connect)
                    existing = discord.utils.get(bot.voice_clients, guild=target.guild)
                    try:
                        if existing and existing.is_connected():
                            if existing.channel and existing.channel.id != target.id:
                                await existing.move_to(target)
                        else:
                            vc = await target.connect()
                            # start én recorder per VC
                            if not getattr(vc, "_listener_started", False):
                                bot.loop.create_task(voice_listener(vc))
                                vc._listener_started = True
                    except Exception as e:
                        print(f"[poll_voice/join] {e}")

            elif action == "leave":
                for vc in list(bot.voice_clients):
                    try:
                        await vc.disconnect()
                    except Exception:
                        pass

            await asyncio.sleep(0.3)  # alltid en liten pause
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
    bot.run(os.getenv("DISCORD_TOKEN"))
