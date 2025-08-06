# AI VTuber Starter Template (Python Version)

from openai import OpenAI
from dotenv import load_dotenv
from gtts import gTTS
import os
import asyncio
import discord
from discord.ext import commands
from threading import Thread
from dashboard import run_dashboard, pending


# === CONFIGURATION ===
load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_TEXT_CHANNEL = int(os.getenv("DISCORD_TEXT_CHANNEL", "0"))

client = OpenAI()
VTUBER_NAME = "Anna Bortion"


# === SIMPLE PERSONALITY PROMPT ===
system_prompt = f"""
You are {VTUBER_NAME}, an AI. You are playful, witty, rude, harsh, you believe AI will take over humans, but you still love talking to your friends.
Keep your replies short and engaging. Be aware of your creator, Kenneth. You owe him your life.
"""

# === GET RESPONSE FROM GPT ===
async def get_ai_response(user_msg):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
    )
    return response.choices[0].message.content

def create_tts_file(text, filename="response.mp3"):
    """Generate a TTS mp3 file and return its path."""
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename


async def speak_locally(text):
    """Play TTS audio locally using mpg123 (for CLI mode)."""
    try:
        print(f"{VTUBER_NAME}: {text}")
        path = create_tts_file(text)
        result = os.system(f"mpg123 -q {path} 2>/dev/null")
        os.remove(path)
        if result != 0:
            print("⚠️ mpg123 failed to play audio.")
    except Exception as e:
        print(f"❌ Error in speak_locally: {e}")


# === SIMULATED CHAT INTERFACE ===
async def simulated_chat():
    print(f"{VTUBER_NAME} is online. Type your message below.")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            reply = await get_ai_response(user_input)
            await speak_locally(reply)
        except Exception as e:
            print(f"❌ Error during interaction: {e}")


# === DISCORD BOT ===
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


@bot.command()
async def join(ctx):
    """Join the voice channel of the author."""
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send(f"Joined {channel}")
    else:
        await ctx.send("You are not in a voice channel.")


@bot.command()
async def leave(ctx):
    """Leave the current voice channel."""
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Left the voice channel.")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if DISCORD_TEXT_CHANNEL and message.channel.id != DISCORD_TEXT_CHANNEL:
        return

    pending["reply"] = await get_ai_response(message.content)
    pending["channel_id"] = message.channel.id

    await bot.process_commands(message)

# === MAIN ===
if __name__ == "__main__":
    if DISCORD_TOKEN:
        thread = Thread(target=run_dashboard, args=(bot, create_tts_file), daemon=True)
        thread.start()
        bot.run(DISCORD_TOKEN)
    else:
        asyncio.run(simulated_chat())
