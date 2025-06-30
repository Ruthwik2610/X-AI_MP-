import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


df = pd.read_csv('clean.csv')


#  homepage_None 0.144444 0.027217
#  fax_+49 (721) 608 4548 0.061111 0.032394
#  phone_ 0.044444 0.022222
#  worksAtProject_id13instance 0.044444 0.033333
#  phone_-0.044444 0.022222
#  fax_+49 (721) 608 6580 0.027778 0.000000
#  publication_ 0.022222 0.011111
#  phone_+49 (721) 608 6586 0.022222 0.011111
#  phone_+49 (721) 608 7362 0.022222 0.011111
#  worksAtProject_id59instance


df.fillna('missing', inplace=True)
label_encoders = {}
for col in ['type', 'phone', 'fax','worksAtProject','publication','homepage']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df['affiliation'])


X = df[['type', 'phone', 'fax','worksAtProject','publication','homepage' ]]  # Key features


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


import lime
import lime.lime_tabular
import numpy as np

# Assume X_train, model, and target_encoder are from your previous code

# Create a LIME explainer for tabular data
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=target_encoder.classes_,
    mode='classification'
)

# Choose an instance to explain (for example, the first test sample)
i = 0
test_instance = X_test.iloc[i].values

# Get explanation for the prediction on this instance
exp = explainer.explain_instance(
    data_row=test_instance,
    predict_fn=model.predict_proba,
    num_features=6  # Number of features to show in explanation
)

# Show explanation in notebook or print text explanation
exp.show_in_notebook(show_table=True)
# Or print text explanation
print(exp.as_list())
