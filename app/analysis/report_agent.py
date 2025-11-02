import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return OpenAI(api_key=api_key)


def _report_model() -> str:
    return os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")


def _pairwise(iterable: Iterable[str]) -> List[Tuple[str, str]]:
    items = list(iterable)
    out: List[Tuple[str, str]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a != b:
                out.append(tuple(sorted((a, b))))
    return out


def _confusion_pairs(cluster_labels: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    # cluster_labels: sign -> label
    label_to_signs: Dict[int, List[str]] = defaultdict(list)
    for sign, lab in cluster_labels.items():
        label_to_signs[int(lab)].append(sign)
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for signs in label_to_signs.values():
        for a, b in _pairwise(signs):
            counts[(a, b)] += 1
    return counts


def _build_context(analysis: Dict) -> Dict:
    # Compute best model by silhouette and confusion pairs per model
    best_model = None
    best_sil = -1.0
    model_summaries = {}
    global_confusions: Dict[Tuple[str, str], int] = defaultdict(int)

    for model, data in analysis.items():
        sil = data.get("silhouette")
        used_k = data.get("used_k")
        pca_ratio = data.get("pca_explained_variance_ratio")
        labels = data.get("cluster_labels", {})
        model_summaries[model] = {
            "silhouette": sil,
            "used_k": used_k,
            "pca_ratio": pca_ratio,
        }
        if isinstance(sil, (int, float)) and sil is not None and sil > best_sil:
            best_sil = float(sil)
            best_model = model
        conf = _confusion_pairs(labels)
        for k, v in conf.items():
            global_confusions[k] += v

    # Top confused pairs overall
    top_confused = sorted(global_confusions.items(), key=lambda x: (-x[1], x[0]))[:10]
    top_confused_serialized = [{"pair": list(k), "count": v} for k, v in top_confused]

    return {
        "best_model": best_model,
        "best_silhouette": best_sil if best_sil >= 0 else None,
        "models": model_summaries,
        "top_confused_pairs": top_confused_serialized,
    }


def generate_final_report(date: str, analysis_report_path: str, output_md_path: str, *, model: str | None = None) -> str:
    with open(analysis_report_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    context = _build_context(analysis)
    client = _client()
    mdl = model or _report_model()

    system = (
        "Eres un analista de resultados de embeddings y clustering. Responde de forma breve, clara y en español, "
        "apoyándote únicamente en los datos proporcionados. El resultado debe ser Markdown bien formateado."
    )

    user = {
        "date": date,
        "questions": [
            "¿Qué embedding separa mejor los signos?",
            "¿Se observa agrupamiento claro en PCA?",
            "¿Qué signos tienden a confundirse?",
            "¿Influye el intérprete o el modelo de embedding más en la separabilidad?",
        ],
        "analysis": analysis,
        "summary_context": context,
    }

    prompt = (
        "Genera un informe en Markdown que responda exactamente estas 4 preguntas, con secciones claras "
        "(usa títulos H3), conclusiones concisas y referencias a métricas (k, silhouette, varianza explicada).\n\n"
        "Preguntas:\n"
        "1. ¿Qué embedding separa mejor los signos?\n"
        "2. ¿Se observa agrupamiento claro en PCA?\n"
        "3. ¿Qué signos tienden a confundirse?\n"
        "4. ¿Influye el intérprete o el modelo de embedding más en la separabilidad?\n\n"
        "Datos:\n" + json.dumps(user, ensure_ascii=False)
    )

    resp = client.chat.completions.create(
        model=mdl,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )

    content = resp.choices[0].message.content or ""
    Path(os.path.dirname(output_md_path)).mkdir(parents=True, exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_md_path
