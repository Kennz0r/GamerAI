# AI VTuber Starter Template (Python Version)

import openai
import pyttsx3
import asyncio
import websockets
import json
import random

# === CONFIGURATION ===
OPENAI_API_KEY = "your_openai_api_key_here"
openai.api_key = OPENAI_API_KEY

VTUBER_NAME = "Neura-chan"

# === TEXT-TO-SPEECH SETUP ===
tts = pyttsx3.init()
tts.setProperty('rate', 180)

# === SIMPLE PERSONALITY PROMPT ===
system_prompt = f"""
You are {VTUBER_NAME}, an AI VTuber. You are playful, witty, and love talking to your fans.
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
    tts.say(text)
    tts.runAndWait()

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
