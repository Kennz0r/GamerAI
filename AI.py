# AI VTuber Starter Template (Python Version)

from openai import OpenAI
from dotenv import load_dotenv
from gtts import gTTS
import os
import asyncio
import websockets
import json
import random


# === CONFIGURATION ===
load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

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

# === TEXT TO SPEECH ===
def speak_text(text):
    try:
        print(f"{VTUBER_NAME}: {text}")
        tts = gTTS(text=text, lang='en')
        tts.save("response.mp3")
        result = os.system("mpg123 -q response.mp3 2>/dev/null")
        os.remove("response.mp3")
        if result != 0:
            print("⚠️ mpg123 failed to play audio.")
    except Exception as e:
        print(f"❌ Error in speak_text: {e}")


# === SIMULATED CHAT INTERFACE ===
async def simulated_chat():
    print(f"{VTUBER_NAME} is online. Type your message below.")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            reply = await get_ai_response(user_input)
            speak_text(reply)
        except Exception as e:
            print(f"❌ Error during interaction: {e}")


# === MAIN ===
if __name__ == "__main__":
    asyncio.run(simulated_chat())
