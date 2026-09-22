"""Diagnostic-only estimate of transferring B evidence into C tools.

This script uses evaluation labels in cross-validation and therefore must not
be used to fit a deployment model or to report final generalization results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage14.evaluation import _oscr  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectories",
        default=("tests/stage14/artifacts/oracle_fold0_abc_smoke/"
                 "trajectories.jsonl"))
    args = parser.parse_args()
    path = ROOT / args.trajectories
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    unknown = np.asarray([int(row["label"] < 0) for row in rows])
    labels = np.asarray([row["label"] for row in rows])
    hypotheses = np.asarray([
        row["trajectory"][-1]["current_hypothesis"] for row in rows])

    base_rows, geometry_rows = [], []
    for row in rows:
        messages = [message for step in row["trajectory"]
                    for message in step["messages"]]
        identity = [message for message in messages
                    if message["sender"] == "identity"]
        geometry = [message for message in messages
                    if message["sender"] == "geometry" and
                    message["message_type"] in
                    {"SUPPORT", "CHALLENGE", "EVIDENCE"}]
        a = identity[0] if identity else {
            "confidence": 0.0, "margin": 0.0, "support_risk": 0.0}
        b = geometry[-1] if geometry else {
            "confidence": 0.0, "margin": 0.0, "support_risk": 0.0,
            "message_type": "ABSTAIN"}
        boundary = next((
            step["revealed_evidence"]["boundary"]
            for step in row["trajectory"]
            if "boundary" in step["revealed_evidence"]), 0.0)
        base_rows.append([
            boundary, a["confidence"], a["margin"], a["support_risk"]])
        geometry_rows.append([
            b["confidence"], b["margin"], b["support_risk"],
            float(b["message_type"] == "SUPPORT"),
            float(b["message_type"] == "CHALLENGE")])

    base = np.asarray(base_rows)
    geometry = np.asarray(geometry_rows)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for name, features in (
        ("boundary_plus_a_report", base),
        ("boundary_a_plus_fused_geometry_report", np.c_[base, geometry]),
    ):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.3, max_iter=2000, class_weight="balanced"))
        score = cross_val_predict(
            model, features, unknown, cv=cv, method="predict_proba")[:, 1]
        print(json.dumps({
            "features": name,
            "diagnostic_cv_auroc": float(roc_auc_score(unknown, score)),
            "diagnostic_cv_oscr": float(_oscr(labels, hypotheses, score)),
        }))

    for index, name in enumerate((
        "b_confidence", "b_margin", "b_prototype_risk",
        "b_support", "b_challenge",
    )):
        values = geometry[:, index]
        auc = float(roc_auc_score(unknown, values))
        print(json.dumps({
            "feature": name,
            "direction_free_auc": max(auc, 1.0-auc),
            "known_mean": float(values[unknown == 0].mean()),
            "unknown_mean": float(values[unknown == 1].mean()),
        }))


if __name__ == "__main__":
    main()
