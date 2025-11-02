import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def analyze_embeddings(embeddings_by_model: Dict[str, np.ndarray], signs: list[str]) -> Dict[str, dict]:
    os.makedirs("outputs", exist_ok=True)
    report: Dict[str, dict] = {}

    for model, X in embeddings_by_model.items():
        n_samples = int(X.shape[0])
        if n_samples == 0:
            report[model] = {
                "error": "no_samples",
                "message": "No embeddings available",
            }
            continue

        # PCA to 2D with safety for small samples
        n_features = int(X.shape[1])
        n_comp = 2 if n_samples >= 2 else 1
        n_comp = max(1, min(n_comp, n_features, n_samples))
        pca = PCA(n_components=n_comp, random_state=42)
        Xp = pca.fit_transform(X)
        if Xp.shape[1] == 1:
            # Pad to 2D for plotting
            X2 = np.hstack([Xp, np.zeros((n_samples, 1), dtype=Xp.dtype)])
            pca_ratio = [float(pca.explained_variance_ratio_[0]), 0.0]
        else:
            X2 = Xp
            pca_ratio = [float(v) for v in pca.explained_variance_ratio_[:2]]

        # Adaptive KMeans: k <= n_samples (and at least 1)
        k = 1 if n_samples < 2 else min(12, n_samples)
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = kmeans.fit_predict(X)

        if k > 1 and len(set(labels)) > 1:
            sil = float(silhouette_score(X, labels))
        else:
            sil = None

        # Plot
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab20", s=80, edgecolor="k")
        for i, sign in enumerate(signs[:n_samples]):
            plt.text(X2[i, 0] + 0.02, X2[i, 1] + 0.02, sign, fontsize=9)
        title_extra = f"k={k}"
        if sil is not None:
            title_extra += f" | Silhouette: {sil:.3f}"
        plt.title(f"PCA(2D) + KMeans - {model} ({title_extra})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        out_path = f"outputs/pca_kmeans_{model}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        report[model] = {
            "n_samples": n_samples,
            "used_k": k,
            "pca_explained_variance_ratio": pca_ratio,
            "silhouette": sil,
            "plot_path": out_path,
            "cluster_labels": {signs[i]: int(labels[i]) for i in range(n_samples)},
        }

    with open("outputs/analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report
