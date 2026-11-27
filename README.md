# Aviation Accident Clustering

A coursework project exploring patterns in historical aviation accidents through two exploratory clustering workflows. The analyses are useful for practicing data preparation, feature engineering, clustering, and interpretation; they are not validated aviation-safety models.

## Analyses

1. **Numeric accident profiles** — K-means using year, occupants aboard, fatalities, fatality rate, and ground fatalities.
2. **Text-derived profiles** — rule-based indicators extracted from accident text fields, followed by clustering.

Cluster labels are algorithmic groupings. They do not establish causal accident types, predict future risk, or measure operational safety.

## Data source and version notes

The dataset is associated with Kaggle's [Airplane Crashes Since 1908](https://www.kaggle.com/datasets/saurograndi/airplane-crashes-since-1908), titled *Airplane Crashes and Fatalities Since 1908*. The dataset page describes a historical collection that originated from a Socrata-hosted source. Check the provider page for the terms that apply to each copy before redistribution.

The CSVs in this repository are coursework-era snapshots. A separately downloaded copy from the Kaggle page contains 5,268 records and uses the `Type` field; the repository's numeric analysis expects `AC Type` in its CSV. The copies therefore differ in schema, and their record-level equivalence has not been established. Reconcile columns, rows, and dates before substituting a different download.

## Repository contents

| File | Purpose |
| --- | --- |
| `Assignment4.py` | Numeric-feature clustering, cluster summaries, and interactive plots |
| `final_explore.py` | Text-derived indicator construction and clustering exploration |
| `Airplane_Crashes_and_Fatalities_Since_1908_20190820105639.csv` | Historical input used by the numeric-feature analysis |
| `airplane_crashes.csv` | Additional coursework data file used by the text exploration |

## Interpretation and limitations

- K-means assigns records based on the selected features and scaling choices; results can change when either changes.
- Accident narratives may be incomplete or inconsistent. Rule-based text indicators reflect the chosen keywords and can miss context or encode overlapping categories.
- Fatalities and aboard counts can be missing or recorded inconsistently. Any analysis should report how missing values and derived rates are handled.
- This dataset is observational and historical. The analyses do not support causal claims or operational recommendations.

## Reproducing the coursework

The repository preserves the original school-project scripts and data snapshots. The scripts expect the historical filenames and column layouts described above, so a newly downloaded dataset may need a documented schema-mapping and validation step before it can be used. Review the input paths and column names in each script before running it.
