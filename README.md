Classification and Explanation with XGBoost and LIME

Project Structure:

project/

├── data/

│ ├── aifbfixed\_complete.n3

│ ├── completeDataset.tsv

│ ├── trainingSet.tsv

│ └── testSet.tsv

├── main.ipynb # Baseline model (78% accuracy)

├── improved.ipynb # Refined model (86% accuracy)

├── clean.csv # Preprocessed dataset for improved model

├── requirements.txt # Required packages

└── README.txt # ← You are here

\-------------------------------------------------------------------------------------------

Prerequisites:

Ensure you have Python 3.8+ and the following packages installed:

pip install pandas numpy rdflib scikit-learn xgboost lime matplotlib seaborn

Alternatively, run: pip install -r requirements.txt

\-------------------------------------------------------------------------------------------

Dataset Setup:

⦁Make sure the following files are present in the data/ folder:

⦁aifbfixed\_complete.n3 – RDF Knowledge Graph

⦁trainingSet.tsv / testSet.tsv – Train/test entity URIs + labels

⦁completeDataset.tsv – For label distribution visualization

\-------------------------------------------------------------------------------------------

Execution Order

Step 1: Run main.ipynb

Implements baseline classification pipeline

Uses all features after variance thresholding

Test Accuracy: ~78%

Includes:

⦁RDF parsing, preprocessing

⦁Feature selection

⦁XGBoost training + evaluation

⦁LIME local explanation

Step 2:

⦁Run improved.ipynb

⦁Builds upon main.ipynb using model-driven feature selection

⦁Uses only features with high importance scores

⦁Input data: clean.csv (filtered dataset with valid persons only)

⦁Test Accuracy: 86%

\-------------------------------------------------------------------------------------------

Running the Notebooks:

Open the notebooks with Jupyter Notebook or VSCode, and run cells sequentially.

Sections include:

⦁Setup and Imports

⦁RDF Graph Parsing and Preprocessing

⦁Data Cleaning and Label Extraction

⦁Feature Engineering + Selection

⦁Model Training using XGBoost

⦁LIME-based Explanation

⦁Class Distribution and Feature Importance Visualization

\-------------------------------------------------------------------------------------------

Key Outputs

⦁Train & Test Accuracy + Classification Report

⦁Visualized Label Distribution

⦁Top 20 Features from XGBoost

⦁LIME Explanation for a sample instance

\-------------------------------------------------------------------------------------------

Highlights of the Pipeline:

⦁Converts RDF triples into tabular format

⦁Uses VarianceThreshold for feature filtering

⦁Handles multi-class classification using XGBoost

⦁Applies LIME for instance-level explanation

⦁Improves accuracy from 78% to 86% with feature importance filtering

\-------------------------------------------------------------------------------------------

Troubleshooting:

⦁Ensure all file paths and formats match expectations

⦁Check .n3 encoding if RDF parsing fails

⦁clean.csv should exist before running improved.ipynb

⦁For best LIME results, use discretize\_continuous=True and ensure valid instances

\-------------------------------------------------------------------------------------------

References:

🔗 LIME Tutorial – Official Docs

https://marcotcr.github.io/lime/tutorials/Lime - basic usage, two class case.html

🔗 XGBoost in Python – DataCamp

https://www.datacamp.com/tutorial/xgboost-in-python

🔗 Feature Selection using XGBoost – Dhanya (2021)

https://medium.com/@dhanyahari07/feature-selection-using-xgboost-f0622fb70c4d

🔗 AIFB Dataset – DataHub

https://datahub.io/dataset/aifb
