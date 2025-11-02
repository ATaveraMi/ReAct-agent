import json
import os
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return OpenAI(api_key=api_key)


def _summary_model() -> str:
    return os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")


def summarize(raw_text: str) -> Dict:
    client = _client()
    model = _summary_model()

    system = (
        "You are a concise summarizer for daily horoscopes. "
        "Return ONLY a JSON object with fields: tone (string), facets (object with love, career, health strings), "
        "key_points (array of short strings), and final_summary (string)."
    )

    user = (
        "Resume el siguiente horóscopo. Devuelve JSON únicamente, sin comentarios extra.\n\n"
        f"TEXTO:\n{raw_text}"
    )

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=800,
    )

    content = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except Exception:
        data = {
            "tone": "",
            "facets": {"love": "", "career": "", "health": ""},
            "key_points": [],
            "final_summary": content.strip(),
        }

    # Ensure expected structure
    data.setdefault("tone", "")
    data.setdefault("facets", {})
    data["facets"].setdefault("love", "")
    data["facets"].setdefault("career", "")
    data["facets"].setdefault("health", "")
    data.setdefault("key_points", [])
    data.setdefault("final_summary", "")
    return data
