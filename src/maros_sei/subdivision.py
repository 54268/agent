from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize

from .metrics import evaluate_subdivision


def _fit_gmm_candidates(
    features: np.ndarray,
    candidates: list[int],
    seed: int,
    sample_size: int,
) -> tuple[int, list[dict[str, float | int]]]:
    rng = np.random.default_rng(seed)
    if len(features) > sample_size:
        subset = features[rng.choice(len(features), size=sample_size, replace=False)]
    else:
        subset = features
    history = []
    for components in candidates:
        model = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            reg_covar=1e-5,
            max_iter=150,
            n_init=2,
            random_state=seed,
        ).fit(subset)
        history.append(
            {
                "components": int(components),
                "bic": float(model.bic(subset)),
                "aic": float(model.aic(subset)),
                "converged": bool(model.converged_),
            }
        )
    selected = min(history, key=lambda item: item["bic"])["components"]
    return int(selected), history


def run_unknown_subdivision(
    *,
    embeddings: np.ndarray,
    cache_true_names: np.ndarray,
    cache_is_true_unknown: np.ndarray,
    total_test_unknown: int,
    unknown_cache_precision: float,
    unknown_cache_recall: float,
    seed: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if len(embeddings) < 4:
        metrics = {
            **evaluate_subdivision(np.asarray([], dtype="U1"), np.asarray([], dtype=np.int64)),
            "backend": "gmm_diag_bic",
            "fit_num_clusters": 0,
            "resolved_num_clusters": 0,
            "unknown_cache_precision": float(unknown_cache_precision),
            "unknown_cache_recall": float(unknown_cache_recall),
            "coverage_of_total_test_unknown": 0.0,
            "selection_history": [],
        }
        return metrics, np.full(len(embeddings), -1, dtype=np.int64), embeddings

    normalized = normalize(embeddings.astype(np.float64), norm="l2")
    scaled = StandardScaler().fit_transform(normalized)
    pca_dim = min(int(config.get("pca_dim", 32)), scaled.shape[1], len(scaled) - 1)
    reduced = PCA(n_components=pca_dim, whiten=True, random_state=seed).fit_transform(scaled)
    max_components = min(int(config.get("max_components", 64)), max(2, len(reduced) // 20))
    candidates = sorted(
        set(
            int(value)
            for value in config.get("candidate_components", [4, 8, 12, 16, 24, 32, 40, 48, 56, 64])
            if 2 <= int(value) <= max_components
        )
    )
    if not candidates:
        candidates = [2]
    selected, history = _fit_gmm_candidates(
        reduced, candidates, seed, int(config.get("selection_sample_size", 4000))
    )
    model = GaussianMixture(
        n_components=selected,
        covariance_type="diag",
        reg_covar=1e-5,
        max_iter=250,
        n_init=3,
        random_state=seed,
    ).fit(reduced)
    labels = model.predict(reduced).astype(np.int64)
    eval_mask = cache_is_true_unknown.astype(bool)
    offline = evaluate_subdivision(cache_true_names[eval_mask], labels[eval_mask])
    metrics = {
        **offline,
        "backend": "gmm_diag_bic",
        "feature_mode": "fused_embedding_pca",
        "pca_dim": int(pca_dim),
        "fit_num_clusters": int(selected),
        "resolved_num_clusters": int(len(np.unique(labels))),
        "unknown_cache_size": int(len(labels)),
        "unknown_cache_precision": float(unknown_cache_precision),
        "unknown_cache_recall": float(unknown_cache_recall),
        "coverage_of_selected_true_unknown": 1.0 if np.any(eval_mask) else 0.0,
        "coverage_of_total_test_unknown": float(eval_mask.sum() / max(total_test_unknown, 1)),
        "selection_history": history,
        "note": "True unknown labels are used only for offline evaluation, never for selecting K or fitting clusters.",
    }
    return metrics, labels, reduced

