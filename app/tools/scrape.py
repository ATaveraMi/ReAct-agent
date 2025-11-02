import asyncio
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup


@dataclass
class ScrapeResult:
    sign: str
    date: str
    interpreter: str
    source_url: str
    raw_text: str


def _interpreter_url(sign: str, interpreter: str) -> Optional[str]:
    s = sign.lower()
    if interpreter == "horoscope.com":
        return f"https://www.horoscope.com/us/horoscopes/general/horoscope-general-daily-{s}.aspx"
    if interpreter == "astrology.com":
        return f"https://www.astrology.com/horoscope/daily/{s}.html"
    return None


async def _scrape_with_browser_use(url: str, sign: str, date: str, interpreter: str) -> Optional[ScrapeResult]:
    if not os.getenv("BROWSER_USE_API_KEY"):
        return None
    try:
        from browser_use import Agent, ChatBrowserUse  # type: ignore
    except Exception:
        return None

    prompt = (
        "Navega a la URL y extrae SOLO el texto principal del horóscopo diario. "
        "Devuelve un JSON con la clave 'raw_text'. No incluyas explicaciones.\n"
        f"URL: {url}\n"
        f"Signo: {sign}\n"
        f"Fecha: {date}\n"
        f"Intérprete: {interpreter}"
    )

    try:
        llm = ChatBrowserUse()
        agent = Agent(task=prompt, llm=llm)
        result = await agent.run()
        text = result if isinstance(result, str) else str(result)
        raw_text = None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "raw_text" in parsed:
                raw_text = str(parsed["raw_text"]).strip()
        except Exception:
            raw_text = text.strip()
        if raw_text:
            return ScrapeResult(
                sign=sign,
                date=date,
                interpreter=interpreter,
                source_url=url,
                raw_text=raw_text,
            )
    except Exception:
        return None
    return None


def _scrape_with_requests(url: str, sign: str, date: str, interpreter: str) -> Optional[ScrapeResult]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        container = soup.find("article") or soup.find("main") or soup.find("div", attrs={"id": "content"}) or soup
        text = container.get_text(" ", strip=True)
        if not text:
            return None
        cleaned = " ".join(text.split())
        if len(cleaned) > 5000:
            cleaned = cleaned[:5000]
        return ScrapeResult(
            sign=sign,
            date=date,
            interpreter=interpreter,
            source_url=url,
            raw_text=cleaned,
        )
    except Exception:
        return None


async def scrape(sign: str, date: str, interpreter: str, *, mode: str = "auto") -> Dict:
    """
    mode: 'auto' | 'browser' | 'requests'
    - auto: usa browser-use si hay API key, si no, requests
    - browser: fuerza browser-use
    - requests: fuerza HTTP requests
    """
    url = _interpreter_url(sign, interpreter)
    if not url:
        return {
            "sign": sign,
            "date": date,
            "interpreter": interpreter,
            "source_url": None,
            "raw_text": "",
            "error": f"Interpreter not supported: {interpreter}",
        }

    use_browser = False
    if mode == "browser":
        use_browser = True
    elif mode == "requests":
        use_browser = False
    else:  # auto
        use_browser = bool(os.getenv("BROWSER_USE_API_KEY"))

    result: Optional[ScrapeResult] = None
    if use_browser:
        result = await _scrape_with_browser_use(url, sign, date, interpreter)
        if not result:
            # fallback silently
            result = await asyncio.to_thread(_scrape_with_requests, url, sign, date, interpreter)
    else:
        result = await asyncio.to_thread(_scrape_with_requests, url, sign, date, interpreter)

    if not result:
        return {
            "sign": sign,
            "date": date,
            "interpreter": interpreter,
            "source_url": url,
            "raw_text": "",
            "error": "Scrape failed",
        }

    return {
        "sign": result.sign,
        "date": result.date,
        "interpreter": result.interpreter,
        "source_url": result.source_url,
        "raw_text": result.raw_text,
    }
