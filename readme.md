# CLOUDSENSE: An Automated Maneuver Recognition and Performance Scoring Framework for DCS World

**CLOUDSENSE** is a complete data processing and machine learning framework designed to analyze high-fidelity flight data from **Digital Combat Simulator (DCS) World**. It transforms raw, complex `.acmi` track files into labeled maneuver datasets, powering an end-to-end system for objective pilot performance analysis and scoring.

## Quick Start & Sample Data

**[⬇️ Download Sample DCS ACMI Data Here](https://drive.google.com/drive/folders/178nooqDx1yNS9VZ01ioOjfjEXe_AC1E9)**

1.  Download a sample `.zip.acmi` file from a DCS mission using the link above.
2.  Place the downloaded file inside the `data/` directory.
3.  Follow the "How to Use" instructions below to process it.

---

## Project Vision: An AI-Powered Debriefing & Scoring System for DCS Pilots

Standard DCS debriefings rely on manual track-file review, which can be time-consuming and subjective. **CLOUDSENSE** aims to be the virtual squadron's AI-powered flight instructor—an objective system that automatically analyzes every aspect of a pilot's performance.

The final goal is to create a **comprehensive flight performance scoring system**. Imagine a post-flight report that doesn't just show your flight path, but scores it with actionable feedback:

*   *"Your High-G Turn scored **8/10 for G-stability** but **5/10 for altitude control**."*
*   *"Split-S maneuver at 14:32 was rated **'Excellent'** for energy management."*
*   *"Trend Identified: Consistent loss of heading during Aileron Rolls."*

This system will provide invaluable, data-driven insights to help virtual pilots master everything from basic flight maneuvers (BFM) to complex air combat maneuvering (ACM), enhancing both individual skill and squadron proficiency.

## The Complete Project Pipeline

The framework supports two primary workflows: **training** a new model and **predicting** with an existing one to generate scores.

**Training Workflow:**
[DCS .acmi File] -> `convert` -> `feature` -> `recog` -> `curate` -> `prepare` -> `train` -> **[Trained CLOUDSENSE Model]**

**Prediction & Scoring Workflow:**
[New DCS .acmi File] -> (Run steps `convert` to `prepare`) -> [New Unlabeled Sequences]
                                                                        |
                                            **[Trained CLOUDSENSE Model]** --+--> [predict_maneuvers.py] -> [Prediction Results] -> **[Future: Performance Scoring Engine]**

## Directory Structure
```
.
├── data/                     # Store raw DCS .acmi data here
├── output/                   # Directory for all processed data
├── models/                   # Store trained CLOUDSENSE model files
├── src/                      # Source code
│   ├── acmi_converter.py
│   ├── feature_engineering.py
│   ├── maneuver_recognition.py
│   ├── curate_ml_data.py
│   ├── prepare_data_for_ml.py
│   ├── train_lstm.py
│   └── predict_maneuvers.py
├── run_pipeline.py           # MASTER SCRIPT to control the workflow
├── README.md
└── requirements.txt
```

## Installation & Environment Setup
1.  Clone this repository.
2.  Create and activate a Python virtual environment.
3.  Install dependencies: `pip install -r requirements.txt`

---

## Workflow 1: Training a CLOUDSENSE Model

This workflow takes raw DCS flight data and produces a trained LSTM model.

### Basic Usage
To process a flight file and train a model from start to finish, run:
```bash
python run_pipeline.py data/your_dcs_flight.zip.acmi output/
```
The script will generate a **unique and consistent** name for this run (e.g., `Run-a1b2c3`) by hashing the input filename. The same input file will always produce the same run name.

### Advanced Control (Re-running Steps)
Use `--start-step` or `--single-step` to control the pipeline.
```bash
# Resume a run at the 'recognition' step
python run_pipeline.py data/your_dcs_flight.zip.acmi output/ --start-step recog
```

### Available Pipeline Steps
| Short Name | Step Description                                         |
| :--------- | :------------------------------------------------------- |
| `convert`  | Converts raw `.acmi` files into partitioned CSVs.        |
| `feature`  | Calculates advanced flight dynamics features.            |
| `recog`    | Applies the hierarchical maneuver recognition engine.    |
| `curate`   | Extracts high-value maneuver clips for ML training.      |
| `prepare`  | Converts curated data into `.npy` sequences for the model. |
| `train`    | Trains the LSTM model and saves the `.h5` model and `.joblib` encoder. |

---

## Workflow 2: Analyzing a Flight with a Trained Model

Once you have a trained model, you can use it to analyze new, unlabeled DCS flights.

### Prerequisite
You must have already completed the `train` step, which generates two essential files, for example:
*   `models/Run-a1b2c3_lstm_model.h5`
*   `models/Run-a1b2c3_lstm_model_encoder.joblib`

### Step 1: Process the New Flight Data
Take your new flight file (e.g., `new_mission.zip.acmi`) and run it through the pipeline, stopping before the `train` step. This converts the raw data into the sequence format (`.npy`).

```bash
# This command runs the first 5 steps for the new file.
python run_pipeline.py data/new_mission.zip.acmi output/ --start-step convert
```
Let's assume this creates the file `output/ml_data/Run-b4c5d6_sequences.npy`.

### Step 2: Run Prediction
Use the `predict_maneuvers.py` script. Provide it with your **trained model** and the **new sequences**.

```bash
python src/predict_maneuvers.py models/Run-a1b2c3_lstm_model.h5 output/ml_data/Run-b4c5d6_sequences.npy
```

### Step 3: Interpret the Results
The script will analyze the new data and print a summary of all maneuvers it identified—the first step toward scoring performance.

**Example Output:**
```
--- Prediction Results ---
Total sequences analyzed: 228031

Summary of maneuvers found:
- Aileron_Roll: 4445 sequences
- Chandelle: 9463 sequences
- Immelmann: 74 sequences
- Sustained_Turn: 247 sequences
- nan: 213802 sequences
```