"""Run EDA on the unified corpus and persist summary tables + figures.

Mirrors notebooks/eda.ipynb but:
  - adapts to the current corpus schema (no ipa/synonyms/antonyms fields)
  - writes structured results (CSV + Markdown) to notebooks/eda_results/
  - writes figures to notebooks/figs/

Run:
    conda run -n study python -m scripts.run_eda
"""

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR

sns.set_theme(context="notebook", style="whitegrid")

REPO = Path(__file__).parent.parent
CORPUS = PROCESSED_DIR / "unified_corpus.jsonl"
FIGS = REPO / "notebooks" / "figs"
OUT = REPO / "notebooks" / "eda_results"
FIGS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    tab = pd.crosstab(a, b)
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = tab.values.sum()
    r, k = tab.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom else np.nan


def main() -> None:
    print(f"Loading {CORPUS} ...")
    docs = load_jsonl(CORPUS)
    df_doc = pd.DataFrame(docs)
    df_doc["senses_count"] = df_doc["senses"].apply(len)
    df_doc["word_len"] = df_doc["word"].str.len()
    df_doc["has_etymology"] = df_doc["etymology"].fillna("").str.strip().astype(bool)
    df_doc["n_related"] = df_doc["related_words"].apply(lambda v: len(v or []))
    print(f"  docs = {len(df_doc):,}")

    df = df_doc.explode("senses", ignore_index=True)
    df = pd.concat([df.drop(columns=["senses"]), df["senses"].apply(pd.Series)], axis=1)
    df["def_len_chars"] = df["definition"].fillna("").str.len()
    df["def_len_tokens"] = df["definition"].fillna("").str.split().apply(len)
    df["n_examples"] = df["examples"].apply(lambda v: len(v) if isinstance(v, list) else 0)
    df["has_example"] = df["n_examples"] > 0
    print(f"  senses = {len(df):,}")

    results: dict[str, pd.DataFrame] = {}

    # §5.1 Scale ────────────────────────────────────────────────
    scale = pd.DataFrame({
        "docs": df_doc["source"].value_counts(),
        "senses": df["source"].value_counts(),
        "unique_words": df_doc.groupby("source")["word"].nunique(),
    }).fillna(0).astype(int)
    scale.loc["TOTAL"] = scale.sum()
    results["scale"] = scale
    print("\n[scale]\n", scale)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    df_doc["source"].value_counts().plot.bar(ax=ax[0], color="steelblue")
    ax[0].set(title="Entries per source", ylabel="docs")
    df["source"].value_counts().plot.bar(ax=ax[1], color="indianred")
    ax[1].set(title="Senses per source", ylabel="senses")
    plt.tight_layout(); plt.savefig(FIGS / "source_counts.png", dpi=150); plt.close()

    # §5.2 POS ──────────────────────────────────────────────────
    pos_tab = pd.crosstab(df["source"], df["part_of_speech"]).fillna(0).astype(int)
    pos_norm = pos_tab.div(pos_tab.sum(axis=1), axis=0)
    results["pos_counts"] = pos_tab
    results["pos_share"] = pos_norm.round(4)
    print("\n[pos_counts]\n", pos_tab)

    pos_norm.plot.bar(stacked=True, figsize=(10, 4), colormap="tab20")
    plt.ylabel("share"); plt.title("POS mix by source")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(FIGS / "pos_by_source.png", dpi=150); plt.close()

    # Polysemy ──────────────────────────────────────────────────
    poly = df_doc.groupby("source")["senses_count"].describe().round(2)
    results["polysemy"] = poly
    print("\n[polysemy]\n", poly)

    fig, ax = plt.subplots(figsize=(8, 4))
    for src, sub in df_doc.groupby("source"):
        ax.hist(sub["senses_count"], bins=range(1, 30), alpha=0.5, label=src)
    ax.set(xlabel="senses per entry", ylabel="# entries", yscale="log",
           title="Polysemy distribution (log y)")
    ax.legend(); plt.tight_layout(); plt.savefig(FIGS / "polysemy.png", dpi=150); plt.close()

    # Definition length & examples ──────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    q99 = df["def_len_tokens"].quantile(0.99)
    sns.boxplot(data=df[df["def_len_tokens"] < q99], x="source", y="def_len_tokens", ax=ax[0])
    ax[0].set(title="Definition length (tokens, 99th-pct clipped)")
    (df.groupby("source")["has_example"].mean()
       .sort_values(ascending=False)
       .plot.bar(ax=ax[1], color="seagreen"))
    ax[1].set(title="Fraction of senses with ≥1 example", ylim=(0, 1))
    plt.tight_layout(); plt.savefig(FIGS / "def_and_examples.png", dpi=150); plt.close()

    deflen = df.groupby("source")["def_len_tokens"].describe().round(2)
    results["def_length"] = deflen
    print("\n[def_length]\n", deflen)

    # Coverage ──────────────────────────────────────────────────
    cov = df_doc.groupby("source")[["has_etymology"]].mean()
    cov["has_related"] = df_doc.groupby("source")["n_related"].apply(lambda s: (s > 0).mean())
    cov["has_example_any"] = df.groupby("source")["has_example"].mean()
    results["coverage"] = cov.round(4)
    print("\n[coverage]\n", cov.round(4))

    cov.plot.bar(figsize=(9, 4)); plt.ylim(0, 1); plt.ylabel("coverage")
    plt.title("Metadata coverage by source"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIGS / "coverage.png", dpi=150); plt.close()

    # Word length ───────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df_doc, x="word_len", hue="source", bins=40, element="step", stat="density")
    plt.title("Word-length distribution by source")
    plt.tight_layout(); plt.savefig(FIGS / "word_len.png", dpi=150); plt.close()
    results["word_len"] = df_doc.groupby("source")["word_len"].describe().round(2)

    # Vocabulary overlap ────────────────────────────────────────
    vocab = {s: set(g["word"].str.lower()) for s, g in df_doc.groupby("source")}
    overlap_rows = [{"source": s, "unique_words": len(v)} for s, v in vocab.items()]
    vocab_df = pd.DataFrame(overlap_rows).set_index("source")
    pair_rows = []
    for a, b in combinations(vocab, 2):
        inter = len(vocab[a] & vocab[b])
        union = len(vocab[a] | vocab[b])
        pair_rows.append({"pair": f"{a} ∩ {b}", "intersection": inter,
                          "union": union, "jaccard": round(inter / union, 4) if union else 0.0})
    pair_df = pd.DataFrame(pair_rows).set_index("pair") if pair_rows else pd.DataFrame()
    results["vocab_sizes"] = vocab_df
    if not pair_df.empty:
        results["vocab_overlap"] = pair_df
    print("\n[vocab_sizes]\n", vocab_df)
    if not pair_df.empty:
        print("\n[vocab_overlap]\n", pair_df)

    try:
        from upsetplot import from_contents, UpSet
        data = from_contents(vocab)
        UpSet(data, show_counts=True).plot()
        plt.suptitle("Vocabulary overlap across sources")
        plt.savefig(FIGS / "vocab_overlap.png", dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        print("upsetplot not installed — skipping UpSet plot")

    # Labels ────────────────────────────────────────────────────
    cat_counts = df_doc["category"].value_counts()
    diff_counts = df_doc["difficulty"].value_counts()
    results["category_counts"] = cat_counts.to_frame("count")
    results["difficulty_counts"] = diff_counts.to_frame("count")
    print("\n[category]\n", cat_counts)
    print("\n[difficulty]\n", diff_counts)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cat_counts.plot.bar(ax=ax[0], color="slateblue"); ax[0].set(title="Category")
    diff_counts.plot.bar(ax=ax[1], color="darkorange"); ax[1].set(title="Difficulty")
    plt.tight_layout(); plt.savefig(FIGS / "labels.png", dpi=150); plt.close()

    diff_by_src = pd.crosstab(df_doc["source"], df_doc["difficulty"], normalize="index").round(4)
    results["difficulty_by_source"] = diff_by_src
    print("\n[difficulty_by_source]\n", diff_by_src)

    # Correlations ──────────────────────────────────────────────
    num = df_doc[["senses_count", "word_len", "n_related"]].copy()
    avg_def = df.groupby(["word", "source"])["def_len_tokens"].mean().reset_index(drop=True)
    avg_ex = df.groupby(["word", "source"])["n_examples"].mean().reset_index(drop=True)
    num["avg_def_tokens"] = avg_def
    num["avg_examples"] = avg_ex
    corr = num.corr(method="spearman").round(3)
    results["spearman_numeric"] = corr
    print("\n[spearman_numeric]\n", corr)

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Spearman correlation (entry-level features)")
    plt.tight_layout(); plt.savefig(FIGS / "corr_numeric.png", dpi=150); plt.close()

    cat_features = ["source", "part_of_speech", "has_example"]
    targets = ["difficulty", "category"]
    rows = []
    for feat in cat_features:
        for tgt in targets:
            rows.append({"feature": feat, "target": tgt,
                         "cramers_v": round(cramers_v(df[feat].astype(str), df[tgt].astype(str)), 4)})
    cramer = pd.DataFrame(rows).pivot(index="feature", columns="target", values="cramers_v")
    results["cramers_v"] = cramer
    print("\n[cramers_v]\n", cramer)

    diff_order = {"beginner": 0, "intermediate": 1, "advanced": 2}
    df_doc["difficulty_ord"] = df_doc["difficulty"].map(diff_order)
    numeric_cols = ["senses_count", "word_len", "n_related"]
    diff_corr = (df_doc[numeric_cols + ["difficulty_ord"]]
                 .corr(method="spearman")["difficulty_ord"]
                 .drop("difficulty_ord").sort_values(key=abs, ascending=False).round(4))
    results["difficulty_spearman"] = diff_corr.to_frame("spearman_vs_difficulty")
    print("\n[difficulty_spearman]\n", diff_corr)

    # §5.1 summary table ────────────────────────────────────────
    summary = pd.DataFrame({
        "docs": df_doc["source"].value_counts(),
        "senses": df["source"].value_counts(),
        "unique_words": df_doc.groupby("source")["word"].nunique(),
        "avg_senses/doc": df_doc.groupby("source")["senses_count"].mean().round(2),
        "avg_def_tokens": df.groupby("source")["def_len_tokens"].mean().round(1),
        "example_rate": df.groupby("source")["has_example"].mean().round(4),
        "etym_rate": df_doc.groupby("source")["has_etymology"].mean().round(4),
        "related_rate": df_doc.groupby("source")["n_related"].apply(lambda s: (s > 0).mean()).round(4),
    })
    results["summary"] = summary
    print("\n[summary]\n", summary)

    # Persist ───────────────────────────────────────────────────
    for name, frame in results.items():
        frame.to_csv(OUT / f"{name}.csv")
    print(f"\n  → wrote {len(results)} CSVs to {OUT}")

    md_path = OUT / "summary.md"
    with md_path.open("w") as fh:
        fh.write("# EDA Summary — WordNet + UrbanDict unified corpus\n\n")
        fh.write(f"Corpus: `{CORPUS.relative_to(REPO)}` · docs = {len(df_doc):,} · senses = {len(df):,}\n\n")
        for name, frame in results.items():
            fh.write(f"## {name}\n\n")
            fh.write(frame.to_markdown() + "\n\n")
    print(f"  → wrote {md_path}")


if __name__ == "__main__":
    main()
