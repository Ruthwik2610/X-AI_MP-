# 📘 LIME Explanation

This document summarizes the LIME (Local Interpretable Model-Agnostic Explanations) output from the model's prediction on the input: **"Andreas Oberweis"**.  
LIME helps interpret **why** a machine learning model made a specific prediction by breaking it down into contributions from each feature (e.g., word).

---

## 🔮 1. Model Prediction Probabilities

The model predicts that the input belongs to one of the following four classes, with associated probabilities:

| Class     | URI                                                                                         | Probability |
|-----------|----------------------------------------------------------------------------------------------|-------------|
| **Class 1** | `http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id1instance` | **16.4%**   |
| **Class 2** | `http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id2instance` | **19.6%**   |
| **Class 3** | `http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id3instance` | **22.1%**   |
| **Class 4** | `http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id4instance` | **41.9%** ✅ _Final Prediction_ |

> 🔹 The model's **final prediction** is **Class 4**, which had the highest probability (41.9%).

---

## 🧠 2. LIME Explanation for Class 2

While Class 4 was the top prediction, LIME can help us understand why the model assigned **19.6%** probability to **Class 2**.

**Input Text:**

"Andreas Oberweis"

markdown
Copy
Edit

### ➕ Feature Contributions for Class 2

| Word       | Weight     | Contribution                                 |
|------------|------------|----------------------------------------------|
| `Andreas`  | -0.0272    | 🔻 Negative – evidence **against** Class 2    |
| `Oberweis` | +0.0095    | 🔺 Positive – evidence **for** Class 2        |

> 📌 Interpretation:  
> The word **"Oberweis"** pushed the model **toward** predicting Class 2, while **"Andreas"** pushed it **away**.

---

## 🧾 Summary

- LIME breaks down the model's decision at the **word level**.
- Even though the final prediction was **Class 4**, this explanation reveals how each word influenced the probability for **Class 2**.
- Such insight is useful for debugging models and understanding the rationale behind predictions.