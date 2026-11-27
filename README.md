# Aviation Accident Clustering

A coursework project exploring patterns in historical aviation accident records. The repository preserves two related analysis tracks:

1. **Numeric profile clustering** — K-means on year, occupants aboard, fatalities, fatality rate, and ground fatalities.
2. **Keyword-derived profile clustering** — rule-based indicators extracted from available text fields, followed by K-means.

These are exploratory analyses, not validated aviation-safety models. The cluster labels are algorithmic groupings and should not be interpreted as causal categories or risk predictions.

## Repository contents

| File | Purpose |
| --- | --- |
| Assignment4.py | Numeric-feature clustering, cluster summaries, and interactive plots |
| final_explore.py | Keyword indicator construction and clustering summary |
| Airplane_Crashes_and_Fatalities_Since_1908_20190820105639.csv | Input for numeric-feature analysis |
| airplane_crashes.csv | Input for keyword-derived analysis |

The CSV files are similar in size and subject, but their equivalence and exact provenance have not been verified. They are retained separately because each script expects a different input file.

## Run locally

Use Python 3.10 or later. Install dependencies:

    python -m pip install -r requirements.txt

Run each analysis from the repository root:

    python Assignment4.py
    python final_explore.py

The first script opens charts interactively and writes cluster_summary.csv, cluster_examples.csv, and clustered_data.csv to the current directory. The second writes cluster_report_k4.csv.

## Interpretation and limitations

- K-means groups records by selected features; it does not discover verified accident causes or estimate future risk.
- In Assignment4.py, fatalities and fatality rate both encode severity, so severity is represented more than once. Missing ground-fatality values are filled with zero, conflating unknown with none. Rates above 1 are clipped rather than investigated.
- In final_explore.py, keyword matches are simple substring rules. They can miss wording, match unrelated fields, overlap, and do not measure semantic text similarity. Missing or undefined fatality rates become zero, and the 0.8 “high fatality” cutoff is hand-set.
- The selected cluster counts (3 and 4) are fixed in the scripts. The first script prints inertia and silhouette scores for several k values, but does not use those results to select k.
- Results depend on data versions, preprocessing, and feature choices. Review the source and data before using or citing any output.

## Data and attribution

The CSV files are included as they were found in the original coursework repository. Their source, license, and version have not been independently confirmed here. Check the dataset provider's terms and document the source before redistributing the data or treating the outputs as reproducible research.

## Related writing

The analyses are related to the author's aviation-accident coursework/report work. The keyword-based script here should not be described as a text-similarity or “twin disasters” system without a separate implementation that actually computes and evaluates text similarity.
