import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return OpenAI(api_key=api_key)


def _embed_models() -> Tuple[str, str]:
    large = os.getenv("OPENAI_EMBED_MODEL_LARGE", "text-embedding-3-large")
    small = os.getenv("OPENAI_EMBED_MODEL_SMALL", "text-embedding-3-small")
    return large, small


def build_embeddings(sign_to_text: Dict[str, str]) -> Dict[str, np.ndarray]:
    client = _client()
    model_large, model_small = _embed_models()

    def embed_batch(model: str, texts: List[str]) -> List[List[float]]:
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    signs_sorted = sorted(sign_to_text.keys())
    texts = [sign_to_text[s] for s in signs_sorted]

    out: Dict[str, np.ndarray] = {}
    for model in [model_large, model_small]:
        vectors = embed_batch(model, texts)
        arr = np.array(vectors, dtype=np.float32)
        out[model] = arr
        # Save CSV/NPY with sign labels
        df = pd.DataFrame(arr)
        df.insert(0, "sign", signs_sorted)
        df.to_csv(f"data/embeddings_{model}.csv", index=False)
        np.save(f"data/embeddings_{model}.npy", arr)
    return out
