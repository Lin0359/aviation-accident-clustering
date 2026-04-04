import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

file_path = "Airplane_Crashes_and_Fatalities_Since_1908_20190820105639.csv"
df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year

cols = ["Date", "Year", "Location", "Operator", "AC Type", "Aboard", "Fatalities", "Ground", "Summary"]
data = df[cols].copy()

for col in ["Year", "Aboard", "Fatalities", "Ground"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=["Year", "Aboard", "Fatalities"])
data = data[data["Aboard"] > 0]
data = data[data["Fatalities"] >= 0]

data["Ground"] = data["Ground"].fillna(0)

data["Fatality_Rate"] = data["Fatalities"] / data["Aboard"]
data["Fatality_Rate"] = data["Fatality_Rate"].clip(0, 1)

data["Ground_Impact"] = (data["Ground"] > 0).astype(int)

features = ["Year", "Aboard", "Fatalities", "Fatality_Rate", "Ground"]
X = data[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
sil_scores = []

k_values = range(2, 7)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

for k, i, s in zip(k_values, inertias, sil_scores):
    print(k, i, s)

plt.figure()
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("k")
plt.ylabel("inertia")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(list(k_values), sil_scores, marker="o")
plt.xlabel("k")
plt.ylabel("silhouette")
plt.tight_layout()
plt.show()

k = 3

model = KMeans(n_clusters=k, random_state=42, n_init=10)
data["Cluster"] = model.fit_predict(X_scaled)

print(data["Cluster"].value_counts())

summary = data.groupby("Cluster").agg(
    count=("Cluster", "size"),
    avg_year=("Year", "mean"),
    avg_aboard=("Aboard", "mean"),
    avg_fatalities=("Fatalities", "mean"),
    avg_rate=("Fatality_Rate", "mean"),
    avg_ground=("Ground", "mean")
).round(2)

print(summary)

examples = (
    data.sort_values(["Cluster", "Fatalities"], ascending=[True, False])
    .groupby("Cluster")
    .head(5)[["Date", "Location", "Operator", "AC Type", "Aboard", "Fatalities", "Ground", "Fatality_Rate", "Cluster"]]
)

print(examples)

plt.figure()
for c in sorted(data["Cluster"].unique()):
    sub = data[data["Cluster"] == c]
    plt.scatter(sub["Aboard"], sub["Fatalities"], label=str(c), alpha=0.6)

plt.xlabel("Aboard")
plt.ylabel("Fatalities")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
for c in sorted(data["Cluster"].unique()):
    sub = data[data["Cluster"] == c]
    plt.scatter(sub["Year"], sub["Fatality_Rate"], label=str(c), alpha=0.6)

plt.xlabel("Year")
plt.ylabel("Fatality_Rate")
plt.legend()
plt.tight_layout()
plt.show()

summary.to_csv("cluster_summary.csv")
examples.to_csv("cluster_examples.csv", index=False)
data.to_csv("clustered_data.csv", index=False)