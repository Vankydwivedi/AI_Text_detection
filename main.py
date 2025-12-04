# main.py

import pandas as pd
from pathlib import Path
from joblib import dump
from src.detector import ChampionDetector


def run_training(
    train_path: str = "data/train.csv",
    test_path: str = "data/test.csv",
    model_path: str = "models/champion_detector.pkl",
    n_splits: int = 10,
    n_repeats: int = 2,
):
    print("=" * 70)
    print("Champion++ v4 – Mercor AI Detection (Local Project)")
    print("=" * 70)

    # -------------------------
    # Load data
    # -------------------------
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train["answer"] = train["answer"].fillna("")
    test["answer"] = test["answer"].fillna("")
    if "topic" in train.columns:
        train["topic"] = train["topic"].fillna("")
    if "topic" in test.columns:
        test["topic"] = test["topic"].fillna("")

    print(
        f"Train {train.shape}, Test {test.shape}, "
        f"Cheating ratio {train['is_cheating'].mean():.3f}"
    )

    # -------------------------
    # Train model
    # -------------------------
    det = ChampionDetector(
        use_embeddings=True,
        svd_components=256,
        random_state=42,
    )

    auc = det.train_ensemble(
        train,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=42,
    )

    # -------------------------
    # Save trained model
    # -------------------------
    Path("models").mkdir(exist_ok=True)
    dump(det, model_path)
    print(f"✅ Model saved to {model_path}")

    # -------------------------
    # Generate predictions
    # -------------------------
    preds = det.predict(train, test)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    sub = pd.DataFrame({"id": test["id"], "is_cheating": preds})
    out_file = out_dir / "championpp_v4.csv"
    sub.to_csv(out_file, index=False, float_format="%.10f")

    print(f"✅ Predictions saved to {out_file}")
    print(f"✅ CV AUC: {auc:.8f}")

    return det, sub, auc


if __name__ == "__main__":
    run_training()
