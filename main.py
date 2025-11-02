import argparse
import asyncio
import os
from datetime import date as dt
from datetime import datetime

from dotenv import load_dotenv

from app.embeddings.analyze import analyze_embeddings
from app.embeddings.build_embeddings import build_embeddings
from app.react_agent import HoroscopeReactAgent
from app.utils.signs import SIGNS
from app.analysis.report_agent import generate_final_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReAct horoscope agent + embeddings + PCA/KMeans")
    parser.add_argument("--date", type=str, default=str(dt.today()), help="Fecha YYYY-MM-DD")
    parser.add_argument(
        "--interpreters",
        nargs="+",
        default=["horoscope.com", "astrology.com"],
        help="Lista de fuentes/intérpretes",
    )
    parser.add_argument(
        "--signs",
        nargs="+",
        default=SIGNS,
        help="Subconjunto de signos a procesar (default: los 12)",
    )
    parser.add_argument(
        "--scrape-mode",
        choices=["auto", "browser", "requests"],
        default="requests",
        help="'auto' usa browser-use si hay API key; 'browser' fuerza browser-use; 'requests' fuerza HTTP",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="Máximo de scrapes concurrentes (para evitar abrir demasiadas pestañas)",
    )
    parser.add_argument(
        "--report-model",
        type=str,
        default=None,
        help="Modelo LLM para el informe final en Markdown (default OPENAI_SUMMARY_MODEL)",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    os.makedirs("data/logs", exist_ok=True)
    load_dotenv()

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = f"data/logs/run_{stamp}.jsonl"

    print(
        f"[RUN] date={args.date} interpreters={args.interpreters} signs={len(args.signs)} "
        f"mode={args.scrape_mode} concurrency={args.max_concurrency}\nLog: {log_path}"
    )

    agent = HoroscopeReactAgent(log_path, max_concurrency=args.max_concurrency, scrape_mode=args.scrape_mode)
    final_per_sign = await agent.run(date=args.date, interpreters=args.interpreters, signs=args.signs)

    # Build embedding inputs: one final summary per sign
    sign_to_text = {sign: (final_per_sign.get(sign, {}).get("final_summary") or "") for sign in args.signs}
    embeddings_by_model = build_embeddings(sign_to_text)

    # Analyze separability
    analysis = analyze_embeddings(embeddings_by_model, signs=args.signs)

    # Generate final report
    analysis_path = "outputs/analysis_report.json"
    report_path = f"outputs/final_analysis_{args.date}.md"
    out_md = generate_final_report(args.date, analysis_path, report_path, model=args.report_model)
    print(f"Informe final: {out_md}")

    print("Listo.")
    print(f"Log: {log_path}")
    print(f"Revisa data/summaries/{args.date}/ y outputs/ para resultados.")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
