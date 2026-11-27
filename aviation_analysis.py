import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description="Explore historical aviation accident records with descriptive clustering.")
    parser.add_argument("--input", required=True, help="Kaggle CSV or ZIP containing one CSV")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--clusters", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    columns = {str(c).strip().lower(): c for c in df.columns}
    required = ["date", "aboard", "fatalities"]
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}; found {list(df.columns)}")

    data = pd.DataFrame({
        "date": pd.to_datetime(df[columns["date"]], errors="coerce"),
        "aboard": pd.to_numeric(df[columns["aboard"]], errors="coerce"),
        "fatalities": pd.to_numeric(df[columns["fatalities"]], errors="coerce"),
    })
    data["year"] = data["date"].dt.year
    data = data.dropna(subset=["year", "aboard", "fatalities"])
    data = data[(data["aboard"] > 0) & (data["fatalities"] >= 0)].copy()
    if len(data) <= args.clusters:
        raise ValueError("Not enough valid rows for the requested cluster count.")

    # Log transforms reduce the leverage of a few exceptionally large events.
    features = pd.DataFrame({
        "year": data["year"],
        "log10_aboard": np.log10(data["aboard"] + 1),
        "log10_fatalities": np.log10(data["fatalities"] + 1),
    })
    scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=args.clusters, random_state=42, n_init=20)
    data["raw_cluster"] = model.fit_predict(scaled)
    order = data.groupby("raw_cluster")["fatalities"].median().sort_values().index.tolist()
    data["cluster"] = data["raw_cluster"].map({raw: i + 1 for i, raw in enumerate(order)})

    summary = data.groupby("cluster").agg(
        records=("cluster", "size"),
        median_year=("year", "median"),
        median_aboard=("aboard", "median"),
        median_fatalities=("fatalities", "median"),
    ).reset_index()
    summary["share_of_valid_records"] = summary["records"] / len(data)
    score = silhouette_score(scaled, data["raw_cluster"], sample_size=min(2000, len(data)), random_state=42)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "cluster_summary.csv", index=False)
    decade_counts = data.assign(decade=(data["year"] // 10) * 10).groupby("decade").size()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [0.9, 1.25]})
    fig.patch.set_facecolor("#f7f8fa")
    left.bar(decade_counts.index.astype(str), decade_counts.values, color="#4477AA")
    left.set(title="Recorded accidents by decade", ylabel="Records in this dataset", xlabel="Decade")
    left.tick_params(axis="x", rotation=55, labelsize=8)
    left.grid(axis="y", alpha=0.2)
    left.set_axisbelow(True)

    colors = ["#4477AA", "#EE9944", "#228833", "#AA3377", "#66CCEE", "#BBBBBB"]
    for cluster in sorted(data["cluster"].unique()):
        group = data[data["cluster"] == cluster]
        right.scatter(group["aboard"], group["fatalities"], s=15, alpha=0.42,
                      color=colors[(cluster - 1) % len(colors)], label=f"Cluster {cluster} (n={len(group):,})",
                      edgecolors="none")
    right.set_xscale("log")
    right.set_yscale("symlog", linthresh=1)
    right.set(title="Exploratory K-means groups", xlabel="People aboard (log scale)", ylabel="Fatalities (symmetric log scale)")
    right.legend(frameon=False, fontsize=8)
    right.grid(alpha=0.18, which="both")
    right.set_axisbelow(True)

    fig.suptitle("Historical aviation accident records", x=0.07, ha="left", fontsize=16, fontweight="bold", color="#172B4D")
    fig.text(0.07, 0.925, f"Kaggle snapshot | {len(data):,} usable records | K={args.clusters} | silhouette={score:.3f}", color="#52657A", fontsize=9)
    fig.text(0.07, 0.015, "Descriptive coursework analysis. Clusters are not causes, safety ratings, or risk predictions. Dataset completeness and reporting change over time.", color="#52657A", fontsize=8)
    plt.tight_layout(rect=[0.03, 0.06, 0.98, 0.88])
    fig.savefig(output / "aviation_accident_patterns.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output / "aviation_accident_patterns.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Input records: {len(df):,}; usable records: {len(data):,}; silhouette: {score:.3f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
