# Explainable AI: AIFB Affiliation Classification

This notebook project studies how to classify researcher affiliations from the AIFB RDF knowledge graph and explain individual predictions. It turns graph facts into tabular features, trains XGBoost classifiers, and uses LIME to show which features influenced a selected prediction.

## Contents

- [Project overview](#project-overview)
- [Repository contents](#repository-contents)
- [Setup](#setup)
- [Run the notebooks](#run-the-notebooks)
- [Recorded results](#recorded-results)
- [Interpretation and limitations](#interpretation-and-limitations)
- [Contributing and support](#contributing-and-support)
- [License](#license)

## Project overview

The baseline notebook parses the RDF graph, prepares person-level features, matches the provided training and test labels, applies variance-based feature filtering, and fits an XGBoost classifier. The refined notebook uses the included `clean.csv`, selects six encoded feature columns, makes a separate train/test split, and fits another XGBoost classifier. Both notebooks generate classification reports and local LIME explanations.

This is a research and learning project, not a deployed prediction service.

## Repository contents

| Path | Purpose |
| --- | --- |
| [`main.ipynb`](main.ipynb) | RDF preprocessing, baseline classifier, evaluation, and explanation |
| [`improved.ipynb`](improved.ipynb) | Refined feature set, classifier, evaluation, and explanation |
| [`data/`](data/) | AIFB RDF graph and supplied TSV datasets |
| [`clean.csv`](clean.csv) | Prepared person-level data used by `improved.ipynb` |
| [`requirements.txt`](requirements.txt) | Pinned core Python packages |
| `LIME_Initial_model.png`, `LIME_improved_model.png` | Saved explanation examples |

## Setup

Use Python 3.10–3.12. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install xgboost seaborn
jupyter lab
```

The notebooks import `xgboost` and `seaborn`, but those packages are not listed in the current `requirements.txt`; the second install command is required until the dependency file is updated. Windows users can activate the virtual environment with `.venv\Scripts\activate`.

## Run the notebooks

1. Open `main.ipynb` from the repository root and run its cells in order. It reads `data/aifbfixed_complete.n3`, `data/trainingSet.tsv`, `data/testSet.tsv`, and `data/completeDataset.tsv`.
2. Open `improved.ipynb` and run its cells in order. It reads the RDF graph and the included root-level `clean.csv`.
3. Review the printed accuracy and classification reports, feature-importance plots, and LIME explanation for a selected test instance.

Keep the notebook working directory at the repository root so relative data paths resolve. Notebook output is illustrative; re-running can produce different explanations or results depending on package versions and random sampling.

## Recorded results

| Notebook | Reported test accuracy in saved output | Evaluation setup |
| --- | ---: | --- |
| `main.ipynb` | 77.78% | Supplied training and test entity lists |
| `improved.ipynb` | 86% | 80/20 split of `clean.csv` with `random_state=42` |

These are results recorded in the committed notebooks, not independently rerun benchmarks. The datasets, features, and split procedures differ, so the figures should not be treated as a controlled comparison of models.

## Interpretation and limitations

- LIME explains one prediction locally; it does not establish causality or guarantee that the model behaves the same way for every person.
- Encoded categorical values and feature selection affect the explanations. Read the feature names and source graph before drawing domain conclusions.
- The refined notebook reports 86% overall accuracy on 28 test rows, but per-class support is small and varies by class. Inspect the full classification report instead of relying on accuracy alone.


## Contributing and support

Open an issue with the notebook name, cell number, expected result, observed result, and Python and package versions. For a change, include a reproducible run and explain any change to data preparation, splitting, or evaluation.
