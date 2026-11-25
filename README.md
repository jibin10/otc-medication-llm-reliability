# Evidence-Grounded OTC Medication Identification

[GitHub repository](https://github.com/jibin10/otc-medication-llm-reliability)

## Project Overview

This project investigates whether compact machine learning models can identify over the counter (OTC) medicines and their active ingredients as reliably as a large language model.

The study compares XGBoost, a hybrid Set Transformer, Gemma 4 31B, and a deterministic retrieval system. It focuses on medicines with similar names, different formulations, shared ingredients, and incomplete product information.

The models were evaluated using 230 selected cases across 153 medication family groups. The hybrid Set Transformer achieved 83.91% joint accuracy, compared with 84.35% for Gemma 4 31B. The study also examines how deterministic evidence verification affects accuracy and appropriate abstention.

This is a research study, not a clinical medication recommendation system.

## Data Sources and Access Requirements

The project uses three main data sources.

- **FDA National Drug Code Directory**  
  Product names, NDC identifiers, dosage forms, strengths, and ingredients.  
  https://open.fda.gov/apis/drug/ndc/

- **FDA Drug Labeling**  
  Structured product labels and supporting drug information.  
  https://open.fda.gov/apis/drug/label/

- **RxNorm**  
  Standardized medication and active ingredient information.  
  https://www.nlm.nih.gov/research/umls/licensedcontent/rxnormfiles.html

### Important Data Access Requirement

**Before accessing or running the notebooks, users must review the applicable dataset usage terms and obtain any required licenses or permissions.**

This project used the full RxNorm release, which requires an NLM/UMLS license agreement. FDA NDC and drug labeling data are publicly available, but their applicable usage conditions must also be followed.

Some notebooks contain saved product names, NDC identifiers, source information, and model outputs. Users are responsible for ensuring that their access and any further redistribution comply with the relevant data agreements.

**The full datasets are not included in this repository.** Users should obtain the required data directly from the official sources above.

## Repository Contents

### Notebooks

| Notebook | Description |
|---|---|
| `2_dataset_feasibility.ipynb` | Explores FDA and RxNorm records, checks data quality, and identifies suitable OTC products |
| `3_benchmark_construction.ipynb` | Constructs the benchmark, prepares reference labels, and separates medication families |
| `4_retrieval_baseline.ipynb` | Implements and evaluates deterministic medication retrieval |
| `6_XGBoost_OTC_Identification.ipynb` | Prepares features, trains XGBoost models, and evaluates development performance |
| `7_Hybrid_Set_Transformer_OTC_Identification.ipynb` | Defines and trains the hybrid Set Transformer and examines its development results |
| `8_Final_Model_Comparison.ipynb` | Compares frozen model predictions and generates final evaluation tables and figures |

The Set Transformer architecture is also defined in `notebooks/model_source.py`. The repository includes its selected training configuration, training history, and aggregate evaluation results.

### Other Files

- `src/hrp/` contains data processing, benchmark construction, retrieval, evidence verification, and evaluation code.
- `tools/` contains supporting scripts for data preparation and model evaluation.
- `tests/` contains tests for the main implementation and scoring methods.
- `results/` contains selected aggregate results, benchmark summaries, and publication figures.
- `hybrid_set_transformer_outputs/` contains selected configuration files and training history.
- `figures/` contains research workflow and architecture diagrams.
- `docs/` contains additional technical documentation.
- `pyproject.toml` contains the main Python project configuration.

## Reproducibility

The notebooks document the main research steps, from data preparation through model evaluation.

To reproduce the study, obtain the required source data, configure the local data paths, and install the relevant Python dependencies.

The full datasets, private benchmark records, model checkpoints, and case level predictions are not distributed in this repository. Reproducing the complete frozen evaluation requires those separately stored artifacts. The repository includes aggregate results and supporting code for reviewing the reported findings.

## Research Limitations

The evaluation uses a selected, source grounded challenge derived from validation data. Its reference answers depend on FDA and RxNorm information and have not undergone complete independent clinical verification.

The findings should not be interpreted as evidence of clinical readiness or general performance across all OTC medicines.