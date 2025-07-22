import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import xgboost as xgb
import os

# --- DATA LOADING ---
@st.cache_data

def load_data():
    df = pd.read_csv('Dataset/Processed Dataset.csv')
    features = {
        'tfidf': np.load('Feature Extraction Files/tfidf_features.npy'),
        'w2v': np.load('Feature Extraction Files/w2v_features.npy'),
        'bert': np.load('Feature Extraction Files/bert_features.npy'),
        'st': np.load('Feature Extraction Files/st_features.npy')
    }
    return df, features

df, features = load_data()

# --- ENCODERS & ADDITIONAL FEATURES ---
encoders = {
    'Priority': LabelEncoder().fit(df['Priority']),
    'Issue_Type': LabelEncoder().fit(df['Issue Type']),
    'Status': LabelEncoder().fit(df['Status']),
    'Component': LabelEncoder().fit(df['Component'])
}
y_priority = encoders['Priority'].transform(df['Priority'])
additional_features = pd.DataFrame({
    'Duration': df['Duration'],
    'Issue_Type': encoders['Issue_Type'].transform(df['Issue Type']),
    'Status_Code': encoders['Status'].transform(df['Status']),
    'Component_Code': encoders['Component'].transform(df['Component'])
})

# --- WORKLOAD MANAGER ---
class WorkloadManager:
    def __init__(self, df):
        self.df = df
        self.user_workload = self._calculate_current_workload()
    def _calculate_current_workload(self):
        return self.df[self.df['Status'] != 'Done'].groupby('Assignee').size()
    def _calculate_task_complexity(self, task_features):
        priority_weight = {'Low': 1, 'Medium': 2, 'High': 3}
        base_complexity = priority_weight[task_features['Priority']]
        duration_factor = min(task_features['Duration'] / 30, 1)
        return base_complexity * (1 + duration_factor)
    def suggest_assignee(self, task_features):
        complexity = self._calculate_task_complexity(task_features)
        workload = self.user_workload.copy()
        all_users = self.df['Assignee'].unique()
        workload = workload.reindex(all_users, fill_value=0)
        workload_scores = workload / workload.max() if workload.max() > 0 else workload
        scores = workload_scores.copy()
        suggested_assignee = scores.idxmin()
        return suggested_assignee, {
            'suggested_assignee': suggested_assignee,
            'workload_score': scores[suggested_assignee],
            'task_complexity': complexity,
            'current_tasks': workload[suggested_assignee]
        }
workload_manager = WorkloadManager(df)

# --- MODEL TRAINING (for demo, train on all data for speed) ---
def get_classification_models():
    models = {}
    scalers = {}
    for feature_name, feature_matrix in features.items():
        # Naive Bayes
        scaler_nb = StandardScaler()
        X_nb = scaler_nb.fit_transform(feature_matrix)
        nb_model = GaussianNB()
        nb_model.fit(X_nb, encoders['Issue_Type'].transform(df['Issue Type']))
        models[f'{feature_name}_nb'] = nb_model
        scalers[f'{feature_name}_nb'] = scaler_nb
        # SVM
        scaler_svm = StandardScaler()
        X_svm = scaler_svm.fit_transform(feature_matrix)
        svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        svm_model.fit(X_svm, encoders['Issue_Type'].transform(df['Issue Type']))
        models[f'{feature_name}_svm'] = svm_model
        scalers[f'{feature_name}_svm'] = scaler_svm
    return models, scalers

def get_priority_models():
    models = {}
    scalers = {}
    for feature_name, feature_matrix in features.items():
        X = np.hstack([feature_matrix, additional_features])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf_model.fit(X_scaled, y_priority)
        models[f'{feature_name}_rf'] = rf_model
        # XGBoost
        xgb_model = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=5, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        xgb_model.fit(X_scaled, y_priority)
        models[f'{feature_name}_xgb'] = xgb_model
        scalers[f'{feature_name}'] = scaler
    return models, scalers

classification_models, classification_scalers = get_classification_models()
priority_models, priority_scalers = get_priority_models()

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Task Management Dashboard", layout="wide")
st.title("AI-Powered Task Management System Dashboard")

st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to section:", [
    "EDA & Data Overview",
    "Task Classification Demo",
    "Priority Prediction Demo",
    "Workload Visualization",
    "Model Performance"
])

# --- EDA & Data Overview ---
if section == "EDA & Data Overview":
    st.header("Exploratory Data Analysis & Data Overview")
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    st.subheader("Priority Distribution")
    st.bar_chart(df['Priority'].value_counts())
    st.subheader("Issue Type Distribution")
    st.bar_chart(df['Issue Type'].value_counts())
    st.subheader("Task Duration by Priority")
    st.box_chart(df[['Priority', 'Duration']].groupby('Priority').describe()['Duration']['mean'])
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# --- Task Classification Demo ---
if section == "Task Classification Demo":
    st.header("Task Classification (Naive Bayes / SVM)")
    st.write("Select a feature type and a task to classify its Issue Type.")
    feature_type = st.selectbox("Feature Type", list(features.keys()))
    task_idx = st.slider("Task Index", 0, len(df)-1, 0)
    task_text = df['Summary'].iloc[task_idx]
    st.write(f"**Task Summary:** {task_text}")
    X = features[feature_type][task_idx].reshape(1, -1)
    # Naive Bayes
    nb_model = classification_models[f'{feature_type}_nb']
    nb_scaler = classification_scalers[f'{feature_type}_nb']
    nb_pred = nb_model.predict(nb_scaler.transform(X))[0]
    nb_label = encoders['Issue_Type'].inverse_transform([nb_pred])[0]
    # SVM
    svm_model = classification_models[f'{feature_type}_svm']
    svm_scaler = classification_scalers[f'{feature_type}_svm']
    svm_pred = svm_model.predict(svm_scaler.transform(X))[0]
    svm_label = encoders['Issue_Type'].inverse_transform([svm_pred])[0]
    st.write(f"**Naive Bayes Prediction:** {nb_label}")
    st.write(f"**SVM Prediction:** {svm_label}")

# --- Priority Prediction Demo ---
if section == "Priority Prediction Demo":
    st.header("Priority Prediction (Random Forest / XGBoost)")
    st.write("Select a feature type and a task to predict its Priority and suggest an assignee.")
    feature_type = st.selectbox("Feature Type", list(features.keys()), key='priority')
    task_idx = st.slider("Task Index", 0, len(df)-1, 0, key='priority_idx')
    X_text = features[feature_type][task_idx].reshape(1, -1)
    add_feat = np.array([
        df['Duration'].iloc[task_idx],
        encoders['Issue_Type'].transform([df['Issue Type'].iloc[task_idx]])[0],
        encoders['Status'].transform([df['Status'].iloc[task_idx]])[0],
        encoders['Component'].transform([df['Component'].iloc[task_idx]])[0]
    ]).reshape(1, -1)
    X_full = np.hstack([X_text, add_feat])
    scaler = priority_scalers[feature_type]
    X_scaled = scaler.transform(X_full)
    # Random Forest
    rf_model = priority_models[f'{feature_type}_rf']
    rf_pred = rf_model.predict(X_scaled)[0]
    rf_label = encoders['Priority'].inverse_transform([rf_pred])[0]
    # XGBoost
    xgb_model = priority_models[f'{feature_type}_xgb']
    xgb_pred = xgb_model.predict(X_scaled)[0]
    xgb_label = encoders['Priority'].inverse_transform([xgb_pred])[0]
    st.write(f"**Random Forest Prediction:** {rf_label}")
    st.write(f"**XGBoost Prediction:** {xgb_label}")
    # Workload suggestion
    task_features = {'Priority': rf_label, 'Duration': df['Duration'].iloc[task_idx]}
    suggested_assignee, assignment_details = workload_manager.suggest_assignee(task_features)
    st.write(f"**Suggested Assignee:** {suggested_assignee}")
    st.write(f"**Workload Score:** {assignment_details['workload_score']:.2f}")
    st.write(f"**Task Complexity:** {assignment_details['task_complexity']:.2f}")

# --- Workload Visualization ---
if section == "Workload Visualization":
    st.header("Workload Distribution Across Assignees")
    workload = workload_manager.user_workload
    st.bar_chart(workload)
    st.write("**Current Workload per Assignee:**")
    st.write(workload)

# --- Model Performance ---
if section == "Model Performance":
    st.header("Model Performance Metrics & Feature Importance")
    st.write("Performance metrics and feature importance for best models.")
    # For demo, show metrics for BERT features
    feature_type = 'bert'
    # Task Classification
    st.subheader("Task Classification (Naive Bayes / SVM)")
    X = features[feature_type]
    y = encoders['Issue_Type'].transform(df['Issue Type'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    nb_model = GaussianNB().fit(X_scaled, y)
    svm_model = SVC(kernel='rbf', probability=True, random_state=42).fit(X_scaled, y)
    nb_pred = nb_model.predict(X_scaled)
    svm_pred = svm_model.predict(X_scaled)
    st.write("**Naive Bayes Classification Report:**")
    st.text(classification_report(y, nb_pred, target_names=encoders['Issue_Type'].classes_))
    st.write("**SVM Classification Report:**")
    st.text(classification_report(y, svm_pred, target_names=encoders['Issue_Type'].classes_))
    # Priority Prediction
    st.subheader("Priority Prediction (Random Forest / XGBoost)")
    X_full = np.hstack([features[feature_type], additional_features])
    y_p = y_priority
    scaler = StandardScaler().fit(X_full)
    X_scaled = scaler.transform(X_full)
    rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42).fit(X_scaled, y_p)
    xgb_model = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=5, use_label_encoder=False, eval_metric='mlogloss', random_state=42).fit(X_scaled, y_p)
    rf_pred = rf_model.predict(X_scaled)
    xgb_pred = xgb_model.predict(X_scaled)
    st.write("**Random Forest Classification Report:**")
    st.text(classification_report(y_p, rf_pred, target_names=encoders['Priority'].classes_))
    st.write("**XGBoost Classification Report:**")
    st.text(classification_report(y_p, xgb_pred, target_names=encoders['Priority'].classes_))
    # Feature Importance
    st.subheader("Feature Importance (Random Forest)")
    importances = rf_model.feature_importances_
    feat_names = [f'Text_{i}' for i in range(features[feature_type].shape[1])] + ['Duration', 'Issue_Type', 'Status', 'Component']
    imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
    imp_df = imp_df.sort_values('Importance', ascending=False).head(20)
    st.bar_chart(imp_df.set_index('Feature'))

# --- REQUIREMENTS ---
# requirements.txt
# streamlit
# numpy
# pandas
# scikit-learn
# xgboost
# matplotlib
# seaborn 