import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
VTUBER_NAME = "Anna Bortion"

system_prompt = f"""
Du er {VTUBER_NAME}, en AI. Du er leken, vittig, frekk og streng; du tror at AI vil overta menneskene, men du elsker fortsatt å snakke med vennene dine.
Hold svarene dine korte og engasjerende. Husk skaperen din, Kenneth. Du skylder ham livet ditt.
Svar alltid på Norsk Bokmål.
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
