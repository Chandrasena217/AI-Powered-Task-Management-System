import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import pickle
import os

class TaskManagementSystem:
    def __init__(self, base_path='D:/AI TASK MANAGEMENT'):
        self.base_path = base_path
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.features = None
        self.df = None
        self.workload_manager = None
        
    def load_data(self):
        """Load and prepare data"""
        # Load dataset
        self.df = pd.read_csv(f'{self.base_path}/Dataset/Processed Dataset.csv')
        
        # Load features
        self.features = {
            'tfidf': np.load(f'{self.base_path}/Feature Extraction Files/tfidf_features.npy'),
            'w2v': np.load(f'{self.base_path}/Feature Extraction Files/w2v_features.npy'),
            'bert': np.load(f'{self.base_path}/Feature Extraction Files/bert_features.npy'),
            'st': np.load(f'{self.base_path}/Feature Extraction Files/st_features.npy')
        }
        
        # Process dates
        self.df['Created'] = pd.to_datetime(self.df['Created'], dayfirst=True)
        self.df['Due Date'] = pd.to_datetime(self.df['Due Date'], dayfirst=True)
        self.df['Duration'] = (self.df['Due Date'] - self.df['Created']).dt.days
        
        # Initialize encoders
        self._initialize_encoders()
        
        # Initialize workload manager
        self.workload_manager = WorkloadManager(self.df)
        
        return self.df, self.features
    
    def _initialize_encoders(self):
        """Initialize and fit label encoders"""
        categorical_columns = ['Priority', 'Issue Type', 'Status', 'Component']
        self.encoders = {
            col: LabelEncoder().fit(self.df[col])
            for col in categorical_columns
        }
    
    def prepare_features(self, text_features, additional_info):
        """Prepare features for prediction"""
        # Encode categorical features
        encoded_features = {
            'Issue_Type': self.encoders['Issue Type'].transform([additional_info['Issue Type']])[0],
            'Status_Code': self.encoders['Status'].transform([additional_info['Status']])[0],
            'Component_Code': self.encoders['Component'].transform([additional_info['Component']])[0]
        }
        
        # Combine features
        additional_features = np.array([
            additional_info['Duration'],
            encoded_features['Issue_Type'],
            encoded_features['Status_Code'],
            encoded_features['Component_Code']
        ]).reshape(1, -1)
        
        return np.hstack([text_features.reshape(1, -1), additional_features])
    
    def train_models(self):
        """Train all models"""
        # Prepare target variables
        y_priority = self.encoders['Priority'].transform(self.df['Priority'])
        y_issue_type = self.encoders['Issue Type'].transform(self.df['Issue Type'])
        
        # Prepare additional features
        additional_features = pd.DataFrame({
            'Duration': self.df['Duration'],
            'Issue_Type': self.encoders['Issue Type'].transform(self.df['Issue Type']),
            'Status_Code': self.encoders['Status'].transform(self.df['Status']),
            'Component_Code': self.encoders['Component'].transform(self.df['Component'])
        })
        
        # Train models for each feature type
        for feature_name, feature_matrix in self.features.items():
            print(f"\nTraining models using {feature_name.upper()} features")
            
            # Combine features
            X = np.hstack([feature_matrix, additional_features])
            
            # Train priority prediction models
            self._train_priority_models(X, y_priority, feature_name)
            
            # Train task classification models
            self._train_classification_models(X, y_issue_type, feature_name)
    
    def _train_priority_models(self, X, y, feature_name):
        """Train priority prediction models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            class_weight='balanced_subsample',
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Train XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            scale_pos_weight=10,
            random_state=42
        )
        xgb_model.fit(X_train_scaled, y_train)
        
        # Store models
        self.models[f"{feature_name}_priority_rf"] = (rf_model, scaler)
        self.models[f"{feature_name}_priority_xgb"] = (xgb_model, scaler)
    
    def _train_classification_models(self, X, y, feature_name):
        """Train task classification models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(X_train_scaled, y_train)
        
        # Train SVM
        svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        svm_model.fit(X_train_scaled, y_train)
        
        # Store models
        self.models[f"{feature_name}_type_nb"] = (nb_model, scaler)
        self.models[f"{feature_name}_type_svm"] = (svm_model, scaler)
    
    def predict_task(self, text_features, additional_info, feature_type='bert'):
        """Make predictions for a new task"""
        # Prepare features
        features = self.prepare_features(text_features, additional_info)
        
        # Get models
        rf_model, rf_scaler = self.models[f"{feature_type}_priority_rf"]
        nb_model, nb_scaler = self.models[f"{feature_type}_type_nb"]
        
        # Scale features
        features_scaled_rf = rf_scaler.transform(features)
        features_scaled_nb = nb_scaler.transform(features)
        
        # Make predictions
        priority_pred = rf_model.predict(features_scaled_rf)[0]
        type_pred = nb_model.predict(features_scaled_nb)[0]
        
        # Get prediction probabilities
        priority_proba = rf_model.predict_proba(features_scaled_rf)[0]
        type_proba = nb_model.predict_proba(features_scaled_nb)[0]
        
        # Convert predictions to labels
        predicted_priority = self.encoders['Priority'].inverse_transform([priority_pred])[0]
        predicted_type = self.encoders['Issue Type'].inverse_transform([type_pred])[0]
        
        # Get assignee suggestion
        task_features = {
            'Priority': predicted_priority,
            'Duration': additional_info['Duration']
        }
        suggested_assignee, assignment_details = self.workload_manager.suggest_assignee(task_features)
        
        return {
            'priority': predicted_priority,
            'priority_confidence': max(priority_proba),
            'issue_type': predicted_type,
            'type_confidence': max(type_proba),
            'suggested_assignee': suggested_assignee,
            'assignment_details': assignment_details
        }
    
    def save_models(self, path='models'):
        """Save trained models and encoders"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save models
        for name, (model, scaler) in self.models.items():
            with open(f'{path}/{name}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open(f'{path}/{name}_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save encoders
        with open(f'{path}/encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
    
    def load_models(self, path='models'):
        """Load trained models and encoders"""
        # Load models
        for name in self.models.keys():
            with open(f'{path}/{name}_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open(f'{path}/{name}_scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            self.models[name] = (model, scaler)
        
        # Load encoders
        with open(f'{path}/encoders.pkl', 'rb') as f:
            self.encoders = pickle.load(f)

class WorkloadManager:
    def __init__(self, df):
        self.df = df
        self.user_workload = self._calculate_current_workload()
    
    def _calculate_current_workload(self):
        """Calculate current workload for each user"""
        return self.df[self.df['Status'] != 'Done'].groupby('Assignee').size()
    
    def _calculate_task_complexity(self, task_features):
        """Calculate task complexity"""
        priority_weight = {'Low': 1, 'Medium': 2, 'High': 3}
        base_complexity = priority_weight[task_features['Priority']]
        duration_factor = min(task_features['Duration'] / 30, 1)
        return base_complexity * (1 + duration_factor)
    
    def suggest_assignee(self, task_features):
        """Suggest best assignee for a task"""
        complexity = self._calculate_task_complexity(task_features)
        workload = self.user_workload.copy()
        
        # Fill missing users
        all_users = self.df['Assignee'].unique()
        workload = workload.reindex(all_users, fill_value=0)
        
        # Calculate workload score
        workload_scores = workload / workload.max() if workload.max() > 0 else workload
        suggested_assignee = workload_scores.idxmin()
        
        return suggested_assignee, {
            'workload_score': workload_scores[suggested_assignee],
            'task_complexity': complexity,
            'current_tasks': workload[suggested_assignee]
        } 