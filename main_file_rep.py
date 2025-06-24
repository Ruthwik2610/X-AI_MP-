# =========================================
# 1. Load RDF Data and Parse Triples
# =========================================
from rdflib import Graph
import pandas as pd

rdf_file = "data/aifbfixed_complete.n3"
graph = Graph()
graph.parse(rdf_file, format='n3')
print("Number of triples in graph:", len(graph))

# Preview first 10 triples
for i, triple in enumerate(graph):
    if i < 10:
        print(triple)

# =========================================
# 2. Load Dataset Files
# =========================================
train_df = pd.read_csv("data/trainingSet.tsv", sep='\t', header=None, names=['person', 'id', 'label_affiliation'])
test_df = pd.read_csv("data/testSet.tsv", sep='\t', header=None, names=['person', 'id', 'label_affiliation'])
full_df = pd.read_csv("data/completeDataset.tsv", sep='\t', header=None, names=['person', 'id', 'label_affiliation'])

print(train_df.head())
print(test_df.head())

# =========================================
# 3. Label Distribution Sanity Check
# =========================================
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Train label distribution:")
print(train_df['label_affiliation'].value_counts(normalize=True))
print("Test label distribution:")
print(test_df['label_affiliation'].value_counts(normalize=True))

# =========================================
# 4. Convert RDF to Entity-Feature Table
# =========================================
from collections import defaultdict

entity_features = defaultdict(dict)

for s, p, o in graph:
    key = f"{p}={o}"
    entity_features[str(s)][key] = 1

features_df = pd.DataFrame.from_dict(entity_features, orient='index').fillna(0).astype(int)
print("Feature matrix shape:", features_df.shape)

# =========================================
# 5. Align Features with Train/Test Sets
# =========================================
train_df = train_df[train_df['person'].isin(features_df.index)]
test_df = test_df[test_df['person'].isin(features_df.index)]

X_train = features_df.loc[train_df['person']]
X_test = features_df.loc[test_df['person']]

# Align y labels
y_train = train_df.set_index('person').loc[X_train.index]['label_affiliation']
y_test = test_df.set_index('person').loc[X_test.index]['label_affiliation']

# =========================================
# 6. Encode Labels
# =========================================
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# =========================================
# 7. Sanity Checks
# =========================================
print("X_train index matches train_df 'person' index:", all(X_train.index == train_df['person']))
print("X_test index matches test_df 'person' index:", all(X_test.index == test_df['person']))
print("\nUnseen labels in test set:", set(y_test) - set(y_train))

# Print shapes
print("X_train shape:", X_train.shape, ", y_train length:", len(y_train_enc))
print("X_test shape:", X_test.shape, ", y_test length:", len(y_test_enc))

# =========================================
# 8. Check for Feature Leakage (correlation)
# =========================================
correlations = X_train.corrwith(pd.Series(y_train_enc, index=X_train.index)).abs()
leak_features = correlations[correlations > 0.5].sort_values(ascending=False)
print("Top correlated features with label:")
print(leak_features.head(10))

# Remove features that directly indicate label
leaky_cols = [col for col in correlations.index if 'affiliation=' in col and any(lbl in col for lbl in y_train.unique())]
X_train_filtered = X_train.drop(columns=leaky_cols)
X_test_filtered = X_test.drop(columns=leaky_cols)

print(f"Features leaking label info: {leaky_cols}")
print(f"Original feature count: {X_train.shape[1]}, after removal: {X_train_filtered.shape[1]}")

# =========================================
# 9. Train Classifier (Random Forest)
# =========================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_filtered, y_train_enc)
y_pred = clf.predict(X_test_filtered)

# =========================================
# 10. Evaluate Model
# =========================================
acc = accuracy_score(y_test_enc, y_pred)
print(f"Test Accuracy (filtered features): {acc:.4f}\n")
print("Classification Report (filtered features):")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_, zero_division=0))
