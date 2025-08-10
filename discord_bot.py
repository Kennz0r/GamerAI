import os
import io
import asyncio
import shutil
import requests
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
import base64
from io import BytesIO
import mimetypes

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

async def _post_vision_update(guild_id: str, image_datauri: str):
    def _post():
        try:
            requests.post(f"{WEB_SERVER_URL}/vision/update",
                          json={"guild_id": guild_id, "image": image_datauri},
                          timeout=5)
        except Exception as e:
            print(f"[vision/update] {e}")
    await asyncio.to_thread(_post)


def _is_img(att):
    ct = (att.content_type or "").lower()
    return ct.startswith("image/") or att.filename.lower().endswith((".png",".jpg",".jpeg",".webp",".gif",".bmp"))

async def _att_to_datauri(att):
    data = await att.read()
    ct = (att.content_type or "").lower() or mimetypes.guess_type(att.filename)[0] or "image/*"
    return f"data:{ct};base64,{base64.b64encode(data).decode()}"

async def _url_to_datauri(url: str) -> str | None:
    def _dl():
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        ct = r.headers.get("Content-Type") or mimetypes.guess_type(url)[0] or "image/*"
        return f"data:{ct};base64,{base64.b64encode(r.content).decode()}"
    try:
        return await asyncio.to_thread(_dl)
    except Exception as e:
        print(f"embed fetch failed: {e}")
        return None

LAST_IMG_B64_BY_GUILD: dict[str, str] = {}

# 1) Dropp hele image-samlingen i on_message()
#    ... fjern løkkene over attachments/embeds og LAST_IMG_B64_BY_GUILD

# 2) send_to_web: ikke aksepter/videresend image_b64
async def send_to_web(channel_id, user_message, user_name, user_id=None, guild_id=None) -> None:
    def _post():
        try:
            payload = {
                "channel_id": channel_id,
                "user_message": user_message,
                "user_name": user_name,
                "user_id": str(user_id) if user_id else None,
                "guild_id": guild_id,
            }
            requests.post(f"{WEB_SERVER_URL}/queue", json=payload, timeout=5)
        except Exception as e:
            print(f"Error sending to web server: {e}")
    await asyncio.to_thread(_post)

# 3) send_audio: ikke legg image i form
async def send_audio(data, user_name, user_id=None, guild_id=None) -> None:
    def _post():
        try:
            files = {"file": ("audio.wav", data, "audio/wav")}
            form = {
                "channel_id": str(DISCORD_TEXT_CHANNEL),
                "user_name": user_name,
                "user_id": str(user_id or ""),
                "guild_id": guild_id or "",
            }
            requests.post(f"{WEB_SERVER_URL}/queue_audio", data=form, files=files, timeout=360)
        except Exception as e:
            print(f"Error sending audio: {e}")
    await asyncio.to_thread(_post)





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
        img = LAST_IMG_B64_BY_GUILD.get(guild_id or "", None)
        await send_audio(audio.file.getvalue(), name, user_id=user_id, guild_id=guild_id, image_b64=img)


async def voice_listener(vc: discord.VoiceClient) -> None:
    """Continuously record short audio snippets and send them for processing."""
    if not HAS_SINKS or not HAS_FFMPEG:
        print("Voice recording not supported; missing sinks or ffmpeg.")
        return

    # shorter chunks → faster STT turnaround
    CHUNK_SEC = float(os.getenv("VOICE_CHUNK_SEC", "1.2"))

    while vc.is_connected():
        # Record raw PCM data (WAV) instead of MP3 to avoid extra compression artifacts
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
            print(f"Recording error: {e}")
            await asyncio.sleep(0.5)


@tasks.loop(seconds=0.5)  # ↓ from 5s
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


@tasks.loop(seconds=0.5)  # ↓ from 5s
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
                print(f"Failed to connect to voice: {e}")
        else:
            print(f"Channel ID {channel_id} not found or not a voice channel.")

    elif action == "leave":
        for vc in list(bot.voice_clients):
            try:
                await vc.disconnect()
            except Exception:
                pass


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    poll_voice.start()
    poll_tts.start()


@bot.event
async def on_message(message: discord.Message):
    try:
        if message.author.bot:
            return
        if DISCORD_TEXT_CHANNEL and message.channel.id != DISCORD_TEXT_CHANNEL:
            return

        image_payload = None
        for att in message.attachments:
            if _is_img(att):
                image_payload = await _att_to_datauri(att)
                break
        if not image_payload:
            for e in message.embeds:
                if getattr(e, "type", None) == "image" and getattr(e, "url", None):
                    image_payload = await _url_to_datauri(e.url)
                    if image_payload:
                        break

        gid = str(message.guild.id) if message.guild else ""
        if image_payload and gid:
            await _post_vision_update(gid, image_payload)

        await bot.process_commands(message)

        content = (message.content or "").lower()
        is_trigger = ((bot.user and bot.user.mentioned_in(message)) or content.startswith(("arne")))
        if is_trigger:
            await send_to_web(
                message.channel.id,
                message.content,
                message.author.display_name,
                user_id=message.author.id,
                guild_id=gid or None,
            )
    except Exception as e:
        print(f"on_message error: {e}")
        
        
        
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)