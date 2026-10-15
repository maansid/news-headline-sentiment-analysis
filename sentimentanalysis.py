import argparse
import io
import os
import re
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
    classification_report, f1_score, roc_auc_score
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, train_test_split
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

OUT_DIR = "outputs_sentiment"
LABELS = {0: "NEGATIVE", 1: "POSITIVE"}
MAX_TEXT_CHARS = 1500


# 1. LOAD AND CLEAN DATA
def clean_text(t):
    t = str(t).replace("\\n", " ").replace("\n", " ")
    t = re.sub(r"https?://\S+", " ", t)
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"[^\w\s']", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def load_data(path):
    lines = []

    with open(path, "rb") as file:
        for raw in file.read().split(b"\n"):
            try:
                lines.append(raw.decode("utf-8"))
            except UnicodeDecodeError:
                lines.append(raw.decode("cp1252"))

    df = pd.read_csv(io.StringIO("\n".join(lines)))

    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["sentiment"])
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    df["sentiment"] = df["sentiment"].astype(int)

    df["headline"] = (
        df["news_title"].fillna("")
        + ". "
        + df["reddit_title"].fillna("")
    ).map(clean_text)

    df["body"] = (
        df["text"].fillna("").str[:MAX_TEXT_CHARS].map(clean_text)
    )

    df["both"] = df["headline"] + " " + df["body"]
    df["n_words"] = df["text"].fillna("").str.split().str.len()

    return df


# 2. EDA GRAPHS
def eda_graphs(df):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    counts = df["sentiment"].map(LABELS).value_counts()

    ax[0].pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=["#4c9f70", "#d9534f"], startangle=90
    )
    ax[0].set_title("Class Balance")

    plot_df = df.assign(Sentiment=df["sentiment"].map(LABELS))

    sns.boxplot(
        data=plot_df, x="Sentiment", y="n_words",
        ax=ax[1], palette=["#d9534f", "#4c9f70"],
        showfliers=False
    )
    ax[1].set_title("Article Length")

    headline_lengths = df["headline"].str.split().str.len()

    sns.histplot(
        data=plot_df, x=headline_lengths,
        hue="Sentiment", bins=25, ax=ax[2],
        palette=["#d9534f", "#4c9f70"]
    )
    ax[2].set_title("Headline Length")
    ax[2].set_xlabel("Words")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_eda.png", dpi=130)
    plt.close()


# 3. POSITIVE AND NEGATIVE WORD LISTS
NEG_WORDS = (
    "kill", "dead", "death", "die", "dies", "died", "crash",
    "attack", "explo", "injur", "violen", "war", "shoot",
    "arrest", "fire", "disaster", "terror", "murder", "victim",
    "tragedy", "collapse", "scandal", "corrupt", "protest",
    "crime", "abuse", "threat", "fear", "bomb", "wound",
    "flee", "destroy", "hurt", "accident", "outbreak", "recall",
    "crisis", "warn", "cheat", "shame", "fraud", "stab",
    "hostage", "famine", "flood", "storm", "hurricane",
    "earthquake", "bloodiest", "deadliest", "suspect", "charged",
    "trial", "convicted", "sentenced", "dangerous", "risk",
    "pollution", "cancer", "poison"
)

POS_WORDS = (
    "win", "won", "save", "rescue", "donat", "free", "cure",
    "help", "hope", "success", "celebrat", "award", "hero",
    "proud", "kind", "charity", "breakthrough", "record",
    "safe", "recover", "reunite", "love", "joy", "thank",
    "wonderful", "achiev", "launch", "improve", "grow", "reduce",
    "ban", "protect", "restor", "fund", "pledge", "wipe",
    "honor", "inspire", "gift"
)


def lexicon_matches(text):
    words = text.split()

    neg = [
        w for w in words
        if any(w.startswith(p) for p in NEG_WORDS)
    ]

    pos = [
        w for w in words
        if any(w.startswith(p) for p in POS_WORDS)
    ]

    return pos, neg


def lexicon_features(texts):
    rows = []

    for t in texts:
        n = max(len(t.split()), 1)
        pos, neg = lexicon_matches(t)

        rows.append([
            np.log1p(len(neg)),
            np.log1p(len(pos)),
            min(100 * len(neg) / n, 30) / 10,
            min(100 * len(pos) / n, 30) / 10
        ])

    return sparse.csr_matrix(np.array(rows))


# 4. TEXT FEATURES AND MODELS
def make_features(use_lexicon=True):
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        max_features=30000
    )

    parts = [("tfidf", tfidf)]

    if use_lexicon:
        parts.append((
            "lex",
            FunctionTransformer(lexicon_features)
        ))

    return FeatureUnion(parts)


def make_models():
    return {
        "Baseline (majority)": DummyClassifier(
            strategy="most_frequent"
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", C=3
        ),

        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced"), cv=3
        ),

        "Naive Bayes": ComplementNB(alpha=0.3),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }


def best_threshold(y, p):
    grid = np.arange(0.2, 0.96, 0.01)

    scores = [
        f1_score(y, (p > t).astype(int), average="macro")
        for t in grid
    ]

    best_index = int(np.argmax(scores))

    return float(grid[best_index]), float(max(scores))


# 5. COMPARE INPUT OPTIONS
def compare_inputs(df):
    skf = StratifiedKFold(
        5, shuffle=True, random_state=42
    )

    rows = []

    for col, cname in [
        ("headline", "Headline only"),
        ("body", "Article text only"),
        ("both", "Headline + text")
    ]:
        for lex in (False, True):
            pipe = Pipeline([
                ("f", make_features(lex)),
                ("clf", LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    C=3
                ))
            ])

            oof = cross_val_predict(
                pipe, df[col], df["sentiment"],
                cv=skf, method="predict_proba"
            )[:, 1]

            _, f1 = best_threshold(df["sentiment"], oof)

            rows.append({
                "Input": cname,
                "col": col,
                "Features": (
                    "TF-IDF + word-list" if lex
                    else "TF-IDF only"
                ),
                "ROC-AUC": roc_auc_score(
                    df["sentiment"], oof
                ),
                "Macro-F1": f1
            })

    res = pd.DataFrame(rows)

    print("\n=== INPUT AND FEATURE COMPARISON ===")
    print(
        res[["Input", "Features", "ROC-AUC", "Macro-F1"]]
        .round(3).to_string(index=False)
    )

    m = res.melt(
        id_vars=["Input", "Features"],
        value_vars=["ROC-AUC", "Macro-F1"],
        var_name="Metric",
        value_name="Score"
    )

    g = sns.catplot(
        data=m, x="Input", y="Score",
        hue="Features", col="Metric",
        kind="bar", palette="Set2",
        height=4.2, aspect=1.1
    )

    g.set(ylim=(0, 1))
    g.fig.suptitle(
        "Effect of Input and Features on Model Quality", y=1.03
    )
    g.savefig(f"{OUT_DIR}/02_input_comparison.png", dpi=130)
    plt.close("all")

    hybrid = res[res["Features"] == "TF-IDF + word-list"]

    return hybrid.sort_values(
        "ROC-AUC", ascending=False
    ).iloc[0]["col"]


# 6. TRAIN AND EVALUATE MODELS
def train_and_evaluate(df, col):
    X, y = df[col], df["sentiment"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )

    skf = StratifiedKFold(
        5, shuffle=True, random_state=42
    )

    rows = []
    fitted = {}
    thr_of = {}
    proba_te = {}

    for name, clf in make_models().items():
        pipe = Pipeline([
            ("f", make_features(True)),
            ("clf", clf)
        ])

        if name.startswith("Baseline"):
            thr = 0.5
        else:
            oof = cross_val_predict(
                pipe, X_tr, y_tr, cv=skf,
                method="predict_proba"
            )[:, 1]

            thr, _ = best_threshold(y_tr, oof)

        pipe.fit(X_tr, y_tr)

        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba > thr).astype(int)

        rows.append({
            "Model": name,
            "Threshold": thr,
            "Test accuracy": accuracy_score(y_te, pred),
            "Test macro-F1": f1_score(
                y_te, pred, average="macro"
            ),
            "ROC-AUC": roc_auc_score(y_te, proba)
        })

        fitted[name] = pipe
        thr_of[name] = thr
        proba_te[name] = proba

    res = pd.DataFrame(rows).sort_values(
        "Test macro-F1", ascending=False
    )

    print("\n=== MODEL COMPARISON ===")
    print(res.round(3).to_string(index=False))

    m = res.drop(columns="Threshold").melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(11, 5))
    sns.barplot(
        data=m, x="Model", y="Score",
        hue="Metric", palette="Set2"
    )
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_model_comparison.png", dpi=130)
    plt.close()

    real_models = res[
        res["Model"] != "Baseline (majority)"
    ]

    best_name = real_models.iloc[0]["Model"]
    best = fitted[best_name]
    thr = thr_of[best_name]

    pred = (proba_te[best_name] > thr).astype(int)

    print(f"\nBest model: {best_name}")
    print(f"Decision threshold: {thr:.2f}")

    print(classification_report(
        y_te, pred,
        target_names=["negative", "positive"],
        zero_division=0
    ))

    fig, ax = plt.subplots(figsize=(5.5, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_te, pred,
        display_labels=["negative", "positive"],
        cmap="Blues", ax=ax
    )

    ax.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_confusion_matrix.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for name, p in proba_te.items():
        if name.startswith("Baseline"):
            continue

        RocCurveDisplay.from_predictions(
            y_te, p, ax=ax,
            name=f"{name} (AUC {roc_auc_score(y_te, p):.2f})"
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC Curves")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_roc_curves.png", dpi=130)
    plt.close()

    lr = fitted["Logistic Regression"]
    tf = lr.named_steps["f"].transformer_list[0][1]

    words = np.array(tf.get_feature_names_out())
    coef = lr.named_steps["clf"].coef_[0][:len(words)]

    keep = np.array([
        not any(w in ENGLISH_STOP_WORDS for w in x.split())
        for x in words
    ])

    w_k, c_k = words[keep], coef[keep]
    order = np.argsort(c_k)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))

    pd.Series(
        c_k[order[-15:]], index=w_k[order[-15:]]
    ).plot.barh(ax=ax[0], color="#4c9f70")

    ax[0].set_title("Words Pushing Towards Positive")

    pd.Series(
        c_k[order[:15]], index=w_k[order[:15]]
    ).plot.barh(ax=ax[1], color="#d9534f")

    ax[1].set_title("Words Pushing Towards Negative")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_top_words.png", dpi=130)
    plt.close()

    return best, best_name, thr, lr, res


# 7. SENTIMENT PREDICTION
class SentimentPredictor:
    def __init__(self, model, threshold, lr_pipeline):
        self.model = model
        self.thr = threshold
        self.lr = lr_pipeline

    def predict(self, text, top_k=5):
        t = clean_text(text)

        p_pos = float(self.model.predict_proba([t])[0][1])

        eps = 1e-6

        def logit(p):
            return np.log((p + eps) / (1 - p + eps))

        score = float(
            1 / (
                1 + np.exp(
                    -(logit(p_pos) - logit(self.thr))
                )
            )
        )

        label = 1 if score > 0.5 else 0

        pos_l, neg_l = lexicon_matches(t)

        tf = self.lr.named_steps["f"].transformer_list[0][1]
        words = np.array(tf.get_feature_names_out())

        coef = self.lr.named_steps["clf"].coef_[0][:len(words)]

        contrib = tf.transform([t]).multiply(coef).tocsr()
        order = np.argsort(contrib.data)

        def ok(w):
            return not any(
                x in ENGLISH_STOP_WORDS for x in w.split()
            )

        pos_t = [
            words[contrib.indices[i]]
            for i in order[::-1]
            if contrib.data[i] > 0
            and ok(words[contrib.indices[i]])
        ]

        neg_t = [
            words[contrib.indices[i]]
            for i in order
            if contrib.data[i] < 0
            and ok(words[contrib.indices[i]])
        ]

        pos_t = [w for w in pos_t if w not in neg_l]
        neg_t = [w for w in neg_t if w not in pos_l]

        def uniq(xs):
            return list(dict.fromkeys(xs))[:top_k]

        confidence = max(score, 1 - score)

        return {
            "label": LABELS[label],
            "confidence": confidence,
            "uncertain": confidence < 0.75,
            "p_positive": score,
            "p_negative": 1 - score,
            "positive_words": uniq(pos_l + pos_t),
            "negative_words": uniq(neg_l + neg_t)
        }

    def show(self, text):
        r = self.predict(text)

        print(f"\nText: {text[:120]}")

        flag = (
            " <- low confidence, treat as unsure"
            if r["uncertain"] else ""
        )

        print(
            f"Sentiment: {r['label']} "
            f"(confidence {r['confidence'] * 100:.1f}%){flag}"
        )

        print(
            f"Score: P(negative)={r['p_negative']:.2f} "
            f"P(positive)={r['p_positive']:.2f}"
        )

        if r["positive_words"]:
            print("Positive cues:", ", ".join(r["positive_words"]))

        if r["negative_words"]:
            print("Negative cues:", ", ".join(r["negative_words"]))

        return r


# 8. DEMO GRAPH
def demo_graph(predictor, sentences):
    rows = [
        {
            "Text": s if len(s) < 55 else s[:52] + "...",
            **predictor.predict(s)
        }
        for s in sentences
    ]

    d = pd.DataFrame(rows)

    plt.figure(figsize=(9, 0.6 * len(d) + 1.5))

    plt.barh(
        d["Text"], d["p_positive"],
        color="#4c9f70", label="positive"
    )

    plt.barh(
        d["Text"], d["p_negative"],
        left=d["p_positive"],
        color="#d9534f", label="negative"
    )

    plt.xlim(0, 1)
    plt.xlabel("Probability")
    plt.title("Demo Predictions")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/07_demo_predictions.png", dpi=130)
    plt.close()


# 9. INTERACTIVE MODE
def interactive(predictor):
    print("\n=== INTERACTIVE MODE: Enter text (q to quit) ===")

    while True:
        t = input("\nYour text> ").strip()

        if t.lower() in {"q", "quit", "exit"}:
            break

        if t:
            predictor.show(t)


# MAIN
def main():
    ap = argparse.ArgumentParser()

    # DATASET FILE PATH
    ap.add_argument(
        "--csv",
        default=r"C:\Users\Mansi\Downloads\archive (1)\Sentiment_dataset.csv"
    )

    ap.add_argument("--interactive", action="store_true")
    ap.add_argument(
        "--load", action="store_true",
        help="Load saved model instead of training"
    )

    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.load:
        bundle = joblib.load(
            f"{OUT_DIR}/sentiment_model.joblib"
        )

        interactive(
            SentimentPredictor(
                bundle["model"],
                bundle["threshold"],
                bundle["lr"]
            )
        )
        return

    # READ THE DATASET
    df = load_data(args.csv)

    print(
        f"Loaded {len(df)} articles | "
        f"positive={(df.sentiment == 1).sum()} "
        f"negative={(df.sentiment == 0).sum()}"
    )

    eda_graphs(df)

    best_col = compare_inputs(df)
    print(f"\nUsing input column: {best_col}")

    best, best_name, thr, lr, _ = train_and_evaluate(
        df, best_col
    )

    predictor = SentimentPredictor(best, thr, lr)

    joblib.dump(
        {
            "model": best,
            "threshold": thr,
            "lr": lr,
            "input": best_col
        },
        f"{OUT_DIR}/sentiment_model.joblib"
    )

    demos = [
        "Local students win national robotics championship and donate prize to charity",
        "Hospital opens free clinic for homeless families in the city",
        "Factory explosion kills dozens and leaves hundreds injured",
        "Government corruption scandal causes massive protests and violence",
        "Scientists discover a new cure that saves thousands of lives",
        "Wildfire destroys hundreds of homes as residents flee",
        "The weather is cloudy today"
    ]

    for s in demos:
        predictor.show(s)

    demo_graph(predictor, demos)

    if args.interactive:
        interactive(predictor)
if __name__ == "__main__":
    main()