# 🔥 FireWatch AI — ML Pipeline

Machine-learning development repository for **FireWatch AI**, an AI-assisted system for contextual classification and analysis of thermal anomalies detected by NASA FIRMS.

This repository contains the complete ML workflow used to develop, evaluate, calibrate, explain, and freeze the final source-classification model used by the FireWatch AI application.

> **Important:** NASA FIRMS provides active-fire / thermal-anomaly observations. FireWatch AI does not treat every FIRMS detection as a confirmed fire or confirmed industrial event. The ML output represents a contextual source classification and industrial detections are reported as **Industrial Fire Candidates**.

---

## Problem Statement

**SIH Problem Statement ID:** SIH26162

**Title:** AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources Using NASA FIRMS, OSM & Satellite Data

**Theme:** Disaster Management  
**Category:** Software

FireWatch AI aims to add contextual intelligence to satellite-detected thermal anomalies by combining:

- NASA FIRMS thermal observations
- Historical thermal activity
- OpenStreetMap geospatial context
- Machine-learning classification
- Explainable AI
- Satellite contextual evidence

The system separates two different questions:

### What is the likely source?

The ML classifier predicts one of three classes:

1. `industrial_fire_candidate`
2. `natural_vegetation_fire`
3. `other_uncertain`

### How does the thermal activity behave over time?

A separate temporal analysis identifies:

- `EPISODIC`
- `RECURRENT`
- `PERSISTENT`

Source classification and temporal behaviour are intentionally kept separate.

---

# Final ML Architecture

The final production classifier is a **CatBoost multiclass classifier** using **26 features**.

```text
NASA FIRMS
    │
    ├── Thermal / observation features
    │
    ├── Historical FIRMS observations
    │       │
    │       └── Causal temporal features
    │
    └── Observation coordinates
            │
            └── OpenStreetMap contextual distances
                        │
                        ▼
                26-Feature Vector
                        │
                        ▼
                 CatBoost Model
                        │
                        ▼
              Probability Calibration
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
       Industrial    Natural      Other /
       Candidate    Vegetation    Uncertain
                        │
                        ├── SHAP explanation
                        ├── Temporal behaviour
                        ├── Thermal-change analysis
                        └── GIS visualization
```

Satellite/Sentinel evidence is retained as **additional contextual evidence** and is not part of the final 26-feature classifier.

---

# Final Feature Set

The frozen production model uses:

**14 FIRMS features + 7 temporal features + 5 OSM features = 26 features**

## 1. NASA FIRMS Features — 14

```text
brightness
bright_t31
frp
scan
track
confidence_code
hour
month
brightness_t31_delta
log_frp
hour_sin
hour_cos
month_sin
month_cos
```

FIRMS confidence values are numerically encoded as:

```text
l -> 0
n -> 1
h -> 2
```

Time-related cyclic features are generated using sine/cosine transformations.

---

## 2. Causal Temporal Features — 7

Historical FIRMS observations are used to characterize previous thermal activity around the current observation.

```text
detections_3d
detections_7d
detections_10d
detections_30d
distinct_active_days_30d
frp_ratio_to_history
active_days_fraction_30d
```

Historical observations are restricted to observations occurring **before the current observation time**, preventing future-data leakage.

The operational search uses approximately:

```text
Spatial radius: <= 1 km
Historical window: previous 30 days
```

---

## 3. OpenStreetMap Context Features — 5

```text
distance_to_strong_industrial_km
distance_to_supporting_industrial_km
distance_to_infrastructure_km
distance_to_vegetation_km
strong_industrial_vs_vegetation_distance_km
```

Raw latitude and longitude are not supplied directly to the classifier.

Coordinates are instead used to retrieve historical and environmental context.

This reduces direct geographic memorization while allowing the model to learn from meaningful contextual features.

> **Coordinates tell the system where to search; derived context tells the model what environment is present.**

---

# Source Classes

The final model predicts three contextual source classes.

### Industrial Fire Candidate

Thermal anomaly whose complete feature pattern is consistent with the industrial candidate class learned during training.

This output should not be interpreted as independent confirmation of an industrial fire.

### Natural / Vegetation Fire

Thermal anomaly whose FIRMS, temporal, and contextual features are more consistent with vegetation/natural-fire examples.

### Other / Uncertain Thermal Anomaly

Used when the observation is better represented by the uncertain/other class than either the industrial or natural class.

This is especially important for avoiding forced binary classification.

---

# Temporal Behaviour

Temporal behaviour is calculated independently from the source classifier.

```text
EPISODIC
RECURRENT
PERSISTENT
```

The implementation uses previous local detections and distinct active days to characterize recurrence.

Keeping temporal behaviour separate is important because a persistent thermal source can still experience an abnormal thermal event.

---

# Thermal Change Analysis

FireWatch also compares a current thermal observation with its recent local historical baseline.

Possible statuses include:

```text
INSUFFICIENT_HISTORY
NORMAL
ELEVATED
ABNORMAL
```

This component helps identify changes in thermal intensity without claiming a physical cause solely from satellite measurements.

For example, an observation can simultaneously be:

```text
Source: Other / Uncertain
Behaviour: PERSISTENT
Thermal Change: ABNORMAL
```

---

# Model Development

Several model families and configurations were evaluated during development.

The repository contains experiments involving:

- Random Forest
- XGBoost
- CatBoost
- CatBoost hyperparameter tuning
- Probability calibration
- Feature-group ablation
- Industrial-class error analysis
- Threshold analysis
- SHAP explainability
- Sensor-transfer auditing

The final production architecture was selected based on the complete evaluation rather than training accuracy alone.

---

# Final Production Model

The frozen classifier uses CatBoost with the following principal configuration:

```text
Iterations:        900
Depth:             7
Learning rate:     0.025
L2 regularization: 5
Random strength:   1
Random seed:       42
```

The final artifacts are stored under:

```text
models/production/source_classifier_v1_final/
```

including:

```text
source_classifier.cbm
source_classifier.joblib
probability_calibrator.joblib
training_medians.joblib
feature_spec.json
FREEZE_MANIFEST.json
```

The production model is frozen as:

```text
source_classifier_v1_final
```

---

# Probability Calibration

Raw CatBoost probabilities are passed through a separate probability-calibration stage.

The calibration component uses multinomial logistic regression applied to transformed CatBoost probability outputs.

This provides more reliable probability estimates for the three source classes.

The final source class is selected from the calibrated class probabilities.

---

# Missing Feature Handling

Missing numerical features are filled using **training-set medians saved during model development**.

The production artifact:

```text
training_medians.joblib
```

ensures that inference uses the same missing-value handling strategy as model development.

---

# Explainable AI

FireWatch uses **SHAP** to explain individual model decisions.

SHAP values are calculated from the CatBoost classifier and indicate which features pushed the model toward or away from a particular source class.

Example influential features can include:

- proximity to industrial context
- proximity to vegetation
- historical detection counts
- FRP relative to historical observations
- thermal characteristics

SHAP explanations refer to the CatBoost model score, while displayed source probabilities are calibrated separately.

---

# Feature Ablation

Feature-group experiments were performed to evaluate different combinations of evidence.

Examples include:

```text
FIRMS only
FIRMS + Temporal
FIRMS + OSM
FIRMS + Sentinel
FIRMS + Temporal + OSM
Full Fusion
```

The final source-classification architecture uses:

```text
FIRMS + Temporal + OSM
```

Satellite/Sentinel information remains available as contextual evidence rather than a production classifier input.

---

# Final Held-Out Evaluation

The frozen model was evaluated using a **temporally held-out 2025 evaluation set containing 20,000 observations**.

## Overall Metrics

| Metric | Result |
|---|---:|
| Accuracy | **90.47%** |
| Balanced Accuracy | **0.7766** |
| Macro F1 | **0.7602** |
| Log Loss | **0.3024** |
| Brier Score | **0.1557** |
| Industrial PR-AUC | **0.7398** |
| Expected Calibration Error | **0.0134** |

## Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Industrial Fire Candidate | 0.6804 | 0.6377 | 0.6584 | 414 |
| Natural / Vegetation Fire | 0.9680 | 0.9325 | 0.9499 | 17,073 |
| Other / Uncertain | 0.6032 | 0.7596 | 0.6724 | 2,513 |

The reported accuracy is the **overall three-class accuracy** and should not be interpreted as industrial-fire detection accuracy.

The evaluation is performed against independently constructed contextual weak labels; therefore the model outputs remain candidate/contextual classifications rather than confirmed physical fire causes.

---

# Sensor Transfer Audit

Historical training data primarily used **S-NPP VIIRS** observations.

The FireWatch live application currently uses:

```text
VIIRS_NOAA21_NRT
```

A recent cross-sensor audit matched observations between S-NPP and NOAA-21 and found approximately:

```text
Source-class argmax agreement: 92.86%
```

This provides useful transfer evidence but is not treated as proof that the two sensors are identical.

---

# Data and Label Construction

NASA FIRMS does not directly provide labels such as:

```text
Industrial Fire
Natural Fire
Other
```

Therefore source labels were constructed using independent contextual/reference evidence.

The repository includes scripts for:

- FIRMS auditing and preprocessing
- temporal feature generation
- OSM extraction and enrichment
- Sentinel evidence extraction
- WorldCover evidence
- industrial reference matching
- source-label construction
- spatial splitting
- model benchmarking
- calibration
- SHAP analysis
- feature ablation
- frozen-model evaluation

Because these labels are contextually constructed rather than direct ground-truth fire-cause observations, FireWatch describes them as **weak labels**.

---

# Data Leakage Prevention

Several precautions were used to reduce leakage.

### Temporal causality

Only historical observations satisfying:

```text
historical_time < current_observation_time
```

are used for temporal features.

### Geographic memorization

Raw latitude and longitude are excluded from the final model.

### Label-generating reference data

Reference information used for constructing weak labels is not directly included as production predictor features.

### Held-out evaluation

The final frozen model was evaluated on a temporally held-out 2025 dataset.

---

# Repository Structure

```text
FireWatch_ML_Final/
│
├── models/
│   ├── experiments/
│   └── production/
│       ├── source_classifier_v1/
│       └── source_classifier_v1_final/
│
├── reports/
│   ├── experiments/
│   └── final/
│
├── scripts/
│   ├── 01_audit_firms.py
│   ├── 02_build_firms_master.py
│   ├── 03_build_temporal_features.py
│   ├── ...
│   ├── 25_build_final_production_model.py
│   ├── ...
│   └── 34_evaluate_frozen_model_2025.py
│
├── data/
│   └── Large/generated datasets excluded from Git
│
├── .gitignore
└── README.md
```

---

# ML Pipeline

The development pipeline broadly follows:

```text
FIRMS Archive
      ↓
Data Audit
      ↓
FIRMS Master Dataset
      ↓
Causal Temporal Features
      ↓
OSM Context Extraction
      ↓
Context / Label Evidence Construction
      ↓
Weak Source Labels
      ↓
Training Dataset
      ↓
Leakage-Aware Splits
      ↓
Model Benchmarking
      ↓
CatBoost Tuning
      ↓
Probability Calibration
      ↓
SHAP + Error Analysis
      ↓
Feature Ablation
      ↓
Final Feature Architecture
      ↓
Frozen Production Model
      ↓
2025 Temporal Holdout Evaluation
```

---

# Important Scripts

The numbered scripts document the development sequence.

Examples:

```text
01_audit_firms.py
02_build_firms_master.py
03_build_temporal_features.py
04_extract_osm_features.py
05_validate_osm_context.py
...
14_build_source_labels.py
15_prepare_training_dataset.py
16_build_spatial_splits.py
17_benchmark_models.py
18_analyze_industrial_errors.py
19_industrial_threshold_analysis.py
20_tune_catboost.py
21_calibrate_catboost.py
22_shap_analysis.py
23_feature_ablation.py
24_final_feature_group_experiment.py
25_build_final_production_model.py
...
34_evaluate_frozen_model_2025.py
```

---

# Dataset Availability

Large datasets are intentionally excluded from this Git repository.

This includes generated/raw files such as:

```text
data/raw/
data/processed/
data/osm/
data/labels/
data/satellite/
```

These files can be large and may also originate from external data providers.

The repository instead preserves the code, model artifacts, experiment summaries, and final evaluation reports required to understand the ML development process.

---

# Integration with FireWatch AI

This repository contains the ML development workflow.

The main **FireWatch AI** application integrates the frozen model into a FastAPI backend.

The operational workflow is:

```text
NASA FIRMS observation
        ↓
Store raw observation
        ↓
Retrieve previous local thermal history
        ↓
Retrieve OSM contextual features
        ↓
Construct same 26-feature vector
        ↓
Frozen CatBoost classifier
        ↓
Probability calibration
        ↓
Persist classification
        ↓
Temporal + thermal-change analysis
        ↓
GIS visualization
```

This separation allows the ML development environment to remain reproducible while the main application focuses on live ingestion, inference, storage, and visualization.

---

# Technology Stack

- Python
- Pandas
- NumPy
- CatBoost
- scikit-learn
- SHAP
- GeoPandas
- OpenStreetMap data
- NASA FIRMS
- Sentinel-2 contextual evidence
- ESA WorldCover contextual/reference evidence
- Parquet

---

# Limitations

FireWatch AI is a decision-support prototype.

Important limitations include:

- FIRMS detections are thermal anomalies, not automatically confirmed fires.
- Industrial outputs are candidate classifications.
- Training labels are contextually constructed weak labels.
- OSM completeness varies geographically.
- Satellite observations may be affected by cloud cover and revisit timing.
- Sensor-transfer performance should continue to be monitored.
- Model probabilities should be interpreted alongside contextual evidence.

The system is designed for **screening, prioritization, and contextual analysis**, not autonomous confirmation of fire cause.

---

# FireWatch AI

**Project:** FireWatch AI  
**Problem Statement:** SIH26162  
**Theme:** Disaster Management  
**ML Model:** CatBoost Multiclass Classifier  
**Production Model:** `source_classifier_v1_final`  
**Features:** 26  
**Source Classes:** 3

The complete FireWatch system combines satellite thermal observations, geospatial context, temporal history, explainable machine learning, and GIS visualization to provide more useful context around detected thermal anomalies.
