## Proyecto: Agente ReAct de Horóscopos + Embeddings + PCA/K-Means

Este proyecto implementa:

- **Parte A (40 pts)**: Un agente ReAct mínimo con dos herramientas:
  - `scrape(sign, date, interpreter)`: obtiene horóscopos desde distintos sitios.
  - `summarize(raw_text)`: resume el texto con estructura: `tone`, `facets` (love, career, health), `key_points`, `final_summary`.
  - Logging en JSONL con `thought`, `action`, `observation`, `final_answer`.
- **Parte B (40 pts)**: Generación de embeddings con **dos modelos de OpenAI** (p. ej. `text-embedding-3-large` y `text-embedding-3-small`), un vector por signo, y análisis de separabilidad con **PCA** y **K-Means** (gráfica + métricas).

Referencia de instalación para Browser Use: ver "Human Quickstart" en `https://docs.browser-use.com/quickstart`.

### Requisitos

- Python 3.12+
- Dependencias en `requirements.txt`
- Variables en `.env`:
  - `BROWSER_USE_API_KEY` (para Browser Use; opcional si usas el fallback HTTP)
  - `OPENAI_API_KEY` (para resúmenes y embeddings)

### Setup (recomendado)

1. Crear y activar venv (con uv, según guía de Browser Use):

```bash
pip install uv
uv venv --python 3.12
source .venv/bin/activate
```

2. Instalar dependencias y chromium para Browser Use:

```bash
uv pip install -r requirements.txt
uvx browser-use install
```

3. Configurar credenciales:

```bash
cp .env.example .env
# Edita .env y agrega tus llaves
```

### Ejecutar end-to-end

Procesa los 12 signos, consolida resúmenes por signo, genera embeddings y realiza PCA+KMeans:

```bash
python main.py --date YYYY-MM-DD --interpreters horoscope.com astrology.com
```

- Salidas:
  - `data/logs/run_*.jsonl`: traza ReAct (thought/action/observation/final_answer)
  - `data/summaries/<date>/<sign>.json`: resumen final por signo
  - `data/embeddings_{model}.csv|.npy`: embeddings por signo
  - `outputs/pca_kmeans_{model}.png`: scatter 2D PCA por modelo
  - `outputs/analysis_report.json`: métricas de clustering

### Notas

- El `scrape` intenta usar Browser Use. Si falla o no hay API key, usa un fallback HTTP con `requests` y `BeautifulSoup` para sitios soportados.
- Los resúmenes y embeddings usan OpenAI. Ajusta modelos en `app/embeddings/build_embeddings.py` y `app/tools/summarize.py`.

### Créditos

- Guía de inicio para Browser Use: `https://docs.browser-use.com/quickstart`
