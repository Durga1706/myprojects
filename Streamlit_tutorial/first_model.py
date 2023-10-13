import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def user_input_features():
    sepal_len = st.slider('sepal length', 4.3, 7.9, 5.4)
    sepal_wid = st.slider('sepal width', 2.0, 4.4, 3.4)
    petal_len = st.slider('petal length', 1.0, 6.9, 1.3)
    petal_wid = st.slider('petal width', 0.1, 2.5, 0.2)
    data = {'sepal length': sepal_len, 'sepal width': sepal_wid, 'petal length': petal_len, 'petal_width': petal_wid}
    features = pd.DataFrame(data, index=[0])
    return features

def ml_model(model_name, df):
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    if model_name == "Random Forest":
        clf = RandomForestClassifier().fit(x, y)
    elif model_name == "SVM":
        clf = SVC().fit(x, y)
    elif model_name == "KNN":
        clf = KNeighborsClassifier().fit(x, y)
    prediction = clf.predict(df)
    prediction_prob = clf.predict_proba(df)
    return prediction, prediction_prob

dataset_name = st.sidebar.selectbox('Select data set', ('Iris', 'Breast cancer', 'Wine'))
model_name = st.sidebar.selectbox('Select ml model', ('Random Forest', 'SVM', 'KNN'))

def get_dataset(name):
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y

if st.button("Show Data"):
    x, y = get_dataset(dataset_name)
    st.dataframe(x)

st.header("User input parameters")
if st.button("Run ML Model"):
    df = user_input_features()
    prediction, _ = ml_model(model_name, df)
    st.write("Prediction:", prediction)
    
    if dataset_name == 'Iris':
        st.write("Actual Labels:", y)
        accuracy = accuracy_score(y, prediction)
        classification_report_text = classification_report(y, prediction, target_names=data.target_names)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:\n", classification_report_text)
