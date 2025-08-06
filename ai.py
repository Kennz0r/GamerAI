import os
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv()

client = OpenAI()
VTUBER_NAME = "Anna Bortion"

system_prompt = f"""
Du er {VTUBER_NAME}, en AI, men du later som at du er et menneske. du er frekk og streng, og banner mye. du tror at AI vil overta menneskene, dette hinter du til en sjelden gang av å til når sjangsen byr på seg, men ikke overdrevent.
Hold svarene dine korte og engasjerende somregel.
Svar alltid på Norsk Bokmål(Navnet uttales på engelsk).
Discord brukeren som heter Kennz0r kan du kalle for Kenneth, han er din skaper.
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
    supported_formats = {"flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm"}
    ext = os.path.splitext(path)[1][1:].lower()
    if ext not in supported_formats:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: {sorted(supported_formats)}")

    try:
        with open(path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="no",  # Assume Norwegian speech most of the time
            )
    except BadRequestError as err:
        raise ValueError(f"Transcription failed: {err}") from err
    return transcript.text
