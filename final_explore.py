import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


INPUT_FILE = "airplane_crashes.csv"
OUTPUT_FILE = "cluster_report_k4.csv"

df = pd.read_csv(INPUT_FILE, encoding="latin1")

text_columns = df.select_dtypes(include="object").columns.tolist()

df["Combined_Text"] = (
    df[text_columns]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.lower()
)

cause_keywords = {
    "Mechanical": ["engine", "mechanical", "failure", "malfunction", "technical", "maintenance"],
    "Weather": ["weather", "storm", "fog", "rain", "snow", "icing", "wind", "thunderstorm"],
    "Pilot": ["pilot", "crew", "error", "misjudgment", "navigation", "controlled flight"],
    "Collision": ["collision", "collided", "midair", "hit", "struck"],
    "Fire_Explosion": ["fire", "explosion", "exploded", "bomb", "blast"],
    "Fuel": ["fuel", "ran out", "shortage"],
    "Terrain": ["mountain", "terrain", "hill", "ground", "crashed into"],
    "Stall": ["stall", "stalled"]
}

scenario_keywords = {
    "Takeoff": ["takeoff", "take off", "taking off"],
    "Landing": ["landing", "landed", "approach"],
    "Cruise": ["cruise", "en route", "mid-flight", "in flight"],
    "Ground": ["ground", "taxi", "runway"]
}

special_keywords = [
    "hijack",
    "terrorist",
    "terrorism",
    "shot down",
    "missile",
    "war",
    "military",
    "sabotage",
    "bomb",
    "explosion"
]

for feature, keywords in cause_keywords.items():
    df[f"Cause_{feature}"] = df["Combined_Text"].apply(
        lambda x: int(any(k in x for k in keywords))
    )

for feature, keywords in scenario_keywords.items():
    df[f"Scenario_{feature}"] = df["Combined_Text"].apply(
        lambda x: int(any(k in x for k in keywords))
    )

df["Special_Case"] = df["Combined_Text"].apply(
    lambda x: int(any(k in x for k in special_keywords))
)

fatality_column = next(
    col for col in df.columns
    if "fatal" in col.lower()
)

aboard_column = next(
    (
        col for col in df.columns
        if "aboard" in col.lower()
    ),
    None
)

df[fatality_column] = pd.to_numeric(
    df[fatality_column],
    errors="coerce"
)

if aboard_column:
    df[aboard_column] = pd.to_numeric(
        df[aboard_column],
        errors="coerce"
    )

    df["Fatality_Rate"] = (
        df[fatality_column] /
        df[aboard_column]
    )
else:
    df["Fatality_Rate"] = df[fatality_column]

df["Fatality_Rate"] = (
    df["Fatality_Rate"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

df["High_Fatality"] = (
    df["Fatality_Rate"] >= 0.8
).astype(int)

feature_columns = [
    "Fatality_Rate",
    "High_Fatality",
    "Cause_Mechanical",
    "Cause_Weather",
    "Cause_Pilot",
    "Cause_Collision",
    "Cause_Fire_Explosion",
    "Cause_Fuel",
    "Cause_Terrain",
    "Cause_Stall",
    "Special_Case",
    "Scenario_Takeoff",
    "Scenario_Landing",
    "Scenario_Cruise",
    "Scenario_Ground"
]

X = df[feature_columns].fillna(0)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

summary = (
    df.groupby("Cluster")[feature_columns]
    .mean()
    .reset_index()
)

summary["Count"] = (
    df.groupby("Cluster")
    .size()
    .values
)

summary["Percentage"] = (
    summary["Count"] /
    len(df)
)

summary = summary[
    [
        "Cluster",
        "Count",
        "Percentage"
    ] + feature_columns
]

summary.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(summary)
print(f"\nSaved to: {OUTPUT_FILE}")