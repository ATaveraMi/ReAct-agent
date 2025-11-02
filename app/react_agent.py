import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from app.tools.scrape import scrape
from app.tools.summarize import summarize
from app.utils.logger import ReactLogger
from app.utils.signs import SIGNS


class HoroscopeReactAgent:
    def __init__(self, log_path: str, *, max_concurrency: int = 2, scrape_mode: str = "requests") -> None:
        self.logger = ReactLogger(log_path, echo_to_stdout=True)
        self.max_concurrency = max(1, int(max_concurrency))
        self.scrape_mode = scrape_mode  # 'auto' | 'browser' | 'requests'

    async def _scrape_one(self, sign: str, date: str, interpreter: str) -> Dict:
        print(f"[SCRAPE] {sign} @ {interpreter}...")
        thought = f"Necesito obtener el horóscopo de {sign} en {interpreter} para {date}."
        self.logger.log(thought=thought, action="scrape", metadata={"sign": sign, "interpreter": interpreter, "date": date})
        result = await scrape(sign, date, interpreter, mode=self.scrape_mode)
        obs = f"Longitud del texto: {len(result.get('raw_text',''))}. Error: {result.get('error')}"
        self.logger.log(observation=obs, metadata={"sign": sign, "interpreter": interpreter})
        if result.get("raw_text"):
            print(f"[SCRAPE ✅] {sign} @ {interpreter} ({len(result['raw_text'])} chars)")
        else:
            print(f"[SCRAPE ❌] {sign} @ {interpreter} -> {result.get('error')}")
        return result

    def _summarize_one(self, raw_text: str, sign: str, source: str) -> Dict:
        print(f"[SUMMARIZE] {sign} from {source}...")
        thought = f"Necesito resumir el texto scraped para {sign} desde {source}."
        self.logger.log(thought=thought, action="summarize", metadata={"sign": sign, "source": source})
        summary = summarize(raw_text)
        obs = f"Resumen OK. Claves: tone/facets/key_points/final_summary"
        self.logger.log(observation=obs, metadata={"sign": sign, "source": source})
        print(f"[SUMMARIZE ✅] {sign} from {source}")
        return summary

    async def run(self, *, date: str, interpreters: List[str], signs: List[str] | None = None) -> Dict[str, Dict]:
        os.makedirs("data/summaries", exist_ok=True)
        out_dir = Path("data/summaries") / date
        out_dir.mkdir(parents=True, exist_ok=True)

        if signs is None:
            signs = SIGNS

        sem = asyncio.Semaphore(self.max_concurrency)

        async def limited_scrape(sign: str, dt: str, interp: str) -> Tuple[str, Dict]:
            async with sem:
                try:
                    result = await asyncio.wait_for(self._scrape_one(sign, dt, interp), timeout=45)
                except asyncio.TimeoutError:
                    msg = "Timeout"
                    self.logger.log(observation=f"Scrape timeout", metadata={"sign": sign, "interpreter": interp})
                    print(f"[SCRAPE ⏱️] {sign} @ {interp} -> Timeout")
                    result = {
                        "sign": sign,
                        "date": dt,
                        "interpreter": interp,
                        "source_url": None,
                        "raw_text": "",
                        "error": msg,
                    }
                return interp, result

        # Scrape with throttling
        per_sign_texts: Dict[str, List[Dict]] = {s: [] for s in signs}
        tasks: List[Tuple[str, asyncio.Task]] = []
        total_jobs = len(signs) * len(interpreters)
        print(f"[START] Scraping {total_jobs} tareas (mode={self.scrape_mode}, concurrency={self.max_concurrency})")
        for sign in signs:
            for interp in interpreters:
                t = asyncio.create_task(limited_scrape(sign, date, interp))
                tasks.append((sign, t))

        for sign, task in tasks:
            interp, result = await task
            if result.get("raw_text"):
                per_sign_texts[sign].append(result)

        # Summarize and consolidate
        final_per_sign: Dict[str, Dict] = {}
        for sign in signs:
            summaries = []
            for item in per_sign_texts[sign]:
                s = self._summarize_one(item["raw_text"], sign, item.get("interpreter", ""))
                s["source_url"] = item.get("source_url")
                s["interpreter"] = item.get("interpreter")
                summaries.append(s)

            combined_text = "\n\n".join([x.get("final_summary", "") for x in summaries if x.get("final_summary")])
            if combined_text:
                consolidated = self._summarize_one(combined_text, sign, "consolidated")
            else:
                consolidated = {"tone": "", "facets": {"love": "", "career": "", "health": ""}, "key_points": [], "final_summary": ""}

            artifact = {
                "sign": sign,
                "date": date,
                "sources": [
                    {
                        "interpreter": i.get("interpreter"),
                        "source_url": i.get("source_url"),
                    }
                    for i in per_sign_texts[sign]
                ],
                "summaries": summaries,
                "final": consolidated,
            }

            with open(out_dir / f"{sign}.json", "w", encoding="utf-8") as f:
                json.dump(artifact, f, ensure_ascii=False, indent=2)

            final_per_sign[sign] = artifact["final"]

        self.logger.log(final_answer=f"Proceso completado para {len(signs)} signos en {date}")
        print(f"[DONE] {len(signs)} signos procesados para {date}")
        return final_per_sign
