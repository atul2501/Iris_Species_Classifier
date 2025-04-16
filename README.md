### 📁 Final File Structure

```
iris_classifier_app/
│
├── Species.py
├── requirements.txt
└── README.md
```

---

### ✅ 1. `Species.py`

```python
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load Iris Data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

# Data and model
df, target_names = load_data()
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

# UI - Sidebar for input
st.sidebar.title("🌸 Input Flower Features")
sepal_length = st.sidebar.slider("Sepal length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

# Predict
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

# Output
st.title("🌼 Iris Species Classifier")
st.success(f"The predicted species is: **{predicted_species}**")
st.info("Adjust the sliders in the sidebar to classify different Iris flowers.")
```

---

### ✅ 2. `requirements.txt`

```txt
streamlit
scikit-learn
pandas
```

---

### ✅ 3. `README.md` (Quick Instructions)

```markdown
# 🌸 One-Click Iris Species Classifier App

This is a Streamlit app that predicts the species of an Iris flower using a Random Forest Classifier. 

## ✅ One-Click Setup & Run

Clone the repo and run the app in one command:

```bash
git clone https://github.com/atul2501/iris_classifier_app.git
cd iris_classifier_app
pip install -r requirements.txt
streamlit run Species.py
```

## 🎯 Features
- Predicts Iris species from sepal/petal inputs
- Built-in interactive sliders
- Beautiful and instant UI via Streamlit

Enjoy predicting flowers with ease! 🌼
```

---

### ✅ Copy-Paste One-Liner for Terminal

```bash
git clone https://github.com/atul2501/iris_classifier_app.git && cd iris_classifier_app && pip install -r requirements.txt && streamlit run Species.py
```
