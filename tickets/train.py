"""Generate data, compare models with cross-validation, train the final classifier and report test results.

    python -m tickets.train            # writes models/ticket_classifier.joblib and reports/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .data import CATEGORIES, held_out_phrasings, make_dataset
from .model import TicketClassifier, candidates, features


def compare(texts, labels, groups, seed=42) -> dict:
    """5-fold cross-validation where a phrasing never appears in both the training and validation fold."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return {name: float(np.mean(cross_val_score(model, texts, labels, groups=groups, cv=cv, scoring="f1_macro")))
            for name, model in candidates(seed).items()}


def train(n=3000, seed=42, model_path="models/ticket_classifier.joblib", report_dir="reports", plot=True) -> dict:
    data = make_dataset(n, seed)
    texts = [d["text"] for d in data]
    cats = [d["category"] for d in data]
    pris = [d["priority"] for d in data]
    langs = [d["lang"] for d in data]
    groups = [d["phrasing"] for d in data]
    held = held_out_phrasings(seed=seed)
    idx_train = np.array([i for i, g in enumerate(groups) if g not in held])
    idx_test = np.array([i for i, g in enumerate(groups) if g in held])
    pick = lambda xs, idx: [xs[i] for i in idx]

    results = {"n_tickets": n, "n_test": len(idx_test), "cv_macro_f1": {}}
    for target, labels in (("category", cats), ("priority", pris)):
        results["cv_macro_f1"][target] = compare(pick(texts, idx_train), pick(labels, idx_train), pick(groups, idx_train), seed)

    final = lambda: Pipeline([("f", features()), ("clf", LogisticRegression(C=8, max_iter=2000))])
    clf = TicketClassifier(category=final(), priority=final())
    clf.category.fit(pick(texts, idx_train), pick(cats, idx_train))
    clf.priority.fit(pick(texts, idx_train), pick(pris, idx_train))

    X_test = pick(texts, idx_test)
    for target, labels, model in (("category", cats, clf.category), ("priority", pris, clf.priority)):
        y_true, y_pred = pick(labels, idx_test), model.predict(X_test)
        results[target] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        }
        for lang in ("en", "de"):
            sel = [i for i, j in enumerate(idx_test) if langs[j] == lang]
            results[target][f"accuracy_{lang}"] = accuracy_score([y_true[i] for i in sel], [y_pred[i] for i in sel])

    proba = clf.category.predict_proba(X_test).max(axis=1)
    confident = proba >= clf.threshold
    y_cat = np.array(pick(cats, idx_test))
    results["routing"] = {
        "auto_routed_share": float(confident.mean()),
        "accuracy_on_auto_routed": float((clf.category.predict(X_test)[confident] == y_cat[confident]).mean()),
    }

    clf.save(model_path)
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if plot:
        _plot_confusion(y_cat, clf.category.predict(X_test), out / "confusion_category.png")
    return results


def _plot_confusion(y_true, y_pred, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=40, ha="right")
    ax.set_yticks(range(len(CATEGORIES)), CATEGORIES)
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Ticket category: confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    r = train(a.n, a.seed)
    print("Cross-validated macro F1 (training split):")
    for target, scores in r["cv_macro_f1"].items():
        print(f"  {target}: " + ", ".join(f"{k} {v:.3f}" for k, v in scores.items()))
    for target in ("category", "priority"):
        t = r[target]
        print(f"Test {target}: accuracy {t['accuracy']:.3f}, macro F1 {t['macro_f1']:.3f} "
              f"(English {t['accuracy_en']:.3f}, German {t['accuracy_de']:.3f})")
    ro = r["routing"]
    print(f"Routing: {ro['auto_routed_share']:.1%} of tickets auto-routed, "
          f"{ro['accuracy_on_auto_routed']:.1%} of those correct; the rest go to a human")


if __name__ == "__main__":
    main()
