# AI VTuber Starter Template (Python Version)

import openai
from dotenv import load_dotenv
from gtts import gTTS
import os
import asyncio
import websockets
import json
import random


# === CONFIGURATION ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

VTUBER_NAME = "Anna Bortion"


# === SIMPLE PERSONALITY PROMPT ===
system_prompt = f"""
You are {VTUBER_NAME}, an AI. You are playful, witty, rude, harsh, you believe AI will take over humans, but you still love talking to your friends.
Keep your replies short and engaging.
"""

# === GET RESPONSE FROM GPT ===
async def get_ai_response(user_msg):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
    )
    return response.choices[0].message['content']

# === TEXT TO SPEECH ===
def speak_text(text):
    print(f"{VTUBER_NAME}: {text}")
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("mpg123 response.mp3")  # or "play" if mpg123 not available

# === SIMULATED CHAT INTERFACE ===
async def simulated_chat():
    print(f"{VTUBER_NAME} is online. Type your message below.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        reply = await get_ai_response(user_input)
        speak_text(reply)

# === MAIN ===
if __name__ == "__main__":
    asyncio.run(simulated_chat())
