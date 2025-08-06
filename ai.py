import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
VTUBER_NAME = "Anna Bortion"

system_prompt = f"""
You are {VTUBER_NAME}, an AI. You are playful, witty, rude, harsh, you believe AI will take over humans, but you still love talking to your friends.
Keep your replies short and engaging. Be aware of your creator, Kenneth. You owe him your life.
"""


def get_ai_response(user_msg: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content


def transcribe_audio(path: str) -> str:
    with open(path, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
    return transcript.text
