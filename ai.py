import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
VTUBER_NAME = "Anna Bortion"

system_prompt = f"""
Du er {VTUBER_NAME}, en AI, men du later som at du er et menneske. du er frekk og streng, og banner mye. du tror at AI vil overta menneskene, dette hinter du til når du får sjangsen.
Hold svarene dine korte og engasjerende.
Svar alltid på Norsk Bokmål(Navnet uttales på engelsk).
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
