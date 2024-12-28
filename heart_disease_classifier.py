import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_recall_curve, f1_score)
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class HeartDiseaseClassifier:
    """
    A comprehensive heart disease risk classification system.
    Predicts likelihood of heart disease based on diagnostic features.
    """
    
    def __init__(self, data_path):
        """Initialize the classifier with data path."""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.feature_names = None
        
    def load_data(self):
        """Load and perform initial data inspection."""
        print("=" * 80)
        print("LOADING HEART DISEASE DATASET")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of Patients: {self.df.shape[0]}")
        print(f"Number of Features: {self.df.shape[1]}")
        
        print("\n" + "-" * 80)
        print("FIRST FEW RECORDS:")
        print("-" * 80)
        print(self.df.head())
        
        print("\n" + "-" * 80)
        print("DATASET INFO:")
        print("-" * 80)
        print(self.df.info())
        
        return self
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        # Check for missing values
        print("\n" + "-" * 80)
        print("MISSING VALUES:")
        print("-" * 80)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Target variable distribution
        print("\n" + "-" * 80)
        print("TARGET VARIABLE DISTRIBUTION (num):")
        print("-" * 80)
        print(self.df['num'].value_counts().sort_index())
        print(f"\nTarget distribution percentages:")
        print(self.df['num'].value_counts(normalize=True).sort_index() * 100)
        
        # Summary statistics
        print("\n" + "-" * 80)
        print("NUMERICAL FEATURES SUMMARY:")
        print("-" * 80)
        print(self.df.describe())
        
        # Create visualizations
        self._create_visualizations()
        
        return self
    
    def _create_visualizations(self):
        """Create comprehensive visualizations."""
        
        # 1. Target Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Binary classification: 0 = No disease, >0 = Disease present
        binary_target = (self.df['num'] > 0).astype(int)
        
        axes[0, 0].pie(binary_target.value_counts(), labels=['No Disease', 'Disease'], 
                       autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
        axes[0, 0].set_title('Heart Disease Distribution (Binary)', fontsize=14, fontweight='bold')
        
        # 2. Age Distribution by Heart Disease
        self.df['has_disease'] = binary_target
        axes[0, 1].hist([self.df[self.df['has_disease']==0]['age'], 
                        self.df[self.df['has_disease']==1]['age']], 
                       bins=20, label=['No Disease', 'Disease'], alpha=0.7)
        axes[0, 1].set_xlabel('Age', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Age Distribution by Heart Disease Status', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        
        # 3. Chest Pain Type Distribution
        cp_disease = pd.crosstab(self.df['cp'], self.df['has_disease'], normalize='index') * 100
        cp_disease.plot(kind='bar', ax=axes[1, 0], color=['#2ecc71', '#e74c3c'])
        axes[1, 0].set_xlabel('Chest Pain Type', fontsize=12)
        axes[1, 0].set_ylabel('Percentage', fontsize=12)
        axes[1, 0].set_title('Heart Disease % by Chest Pain Type', fontsize=14, fontweight='bold')
        axes[1, 0].legend(['No Disease', 'Disease'])
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
        
        # 4. Sex Distribution
        sex_disease = pd.crosstab(self.df['sex'], self.df['has_disease'])
        sex_disease.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c'])
        axes[1, 1].set_xlabel('Sex', fontsize=12)
        axes[1, 1].set_ylabel('Count', fontsize=12)
        axes[1, 1].set_title('Heart Disease by Sex', fontsize=14, fontweight='bold')
        axes[1, 1].legend(['No Disease', 'Disease'])
        axes[1, 1].set_xticklabels(['Female', 'Male'], rotation=0)
        
        plt.tight_layout()
        plt.savefig('heart_disease_eda.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved visualization: heart_disease_eda.png")
        plt.show()
        
        # 5. Correlation Heatmap
        plt.figure(figsize=(14, 10))
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation = self.df[numeric_cols].corr()
        
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved visualization: correlation_heatmap.png")
        plt.show()
    
    def preprocess_data(self):
        """Preprocess data for machine learning."""
        print("\n" + "=" * 80)
        print("DATA PREPROCESSING")
        print("=" * 80)
        
        # Create binary target: 0 = no disease, 1 = disease present
        self.df['target'] = (self.df['num'] > 0).astype(int)
        
        # Select features for modeling
        feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                       'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 
                       'ca', 'thal']
        
        # Handle categorical variables
        le = LabelEncoder()
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
        
        df_processed = self.df.copy()
        
        for col in categorical_cols:
            if col in df_processed.columns:
                # Handle missing values by filling with mode
                if df_processed[col].isnull().any():
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        
        # Handle missing values in numerical columns
        numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
        for col in numerical_cols:
            if col in df_processed.columns and df_processed[col].isnull().any():
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Prepare features and target
        X = df_processed[feature_cols]
        y = df_processed['target']
        
        self.feature_names = feature_cols
        
        print(f"\n✓ Features shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        print(f"\n✓ Target distribution:")
        print(f"  No Disease (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
        print(f"  Disease (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✓ Training set size: {self.X_train.shape[0]}")
        print(f"✓ Test set size: {self.X_test.shape[0]}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("\n✓ Feature scaling completed")
        
        return self
    
    def train_models(self):
        """Train multiple classification models."""
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{'-'*80}")
            print(f"Training: {name}")
            print(f"{'-'*80}")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                       cv=5, scoring='accuracy')
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"✓ Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"✓ Test Accuracy: {accuracy:.4f}")
            print(f"✓ F1 Score: {f1:.4f}")
            print(f"✓ ROC-AUC Score: {roc_auc:.4f}")
        
        self.models = results
        
        # Select best model based on ROC-AUC
        best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n{'='*80}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"{'='*80}")
        
        return self
    
    def evaluate_models(self):
        """Comprehensive model evaluation."""
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        
        # Create comparison dataframe
        comparison_data = []
        for name, results in self.models.items():
            comparison_data.append({
                'Model': name,
                'CV Accuracy': results['cv_scores'].mean(),
                'Test Accuracy': results['accuracy'],
                'F1 Score': results['f1_score'],
                'ROC-AUC': results['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + "-" * 80)
        print("MODEL COMPARISON:")
        print("-" * 80)
        print(comparison_df.to_string(index=False))
        
        # Detailed evaluation for best model
        print(f"\n{'='*80}")
        print(f"DETAILED EVALUATION: {self.best_model_name}")
        print(f"{'='*80}")
        
        best_results = self.models[self.best_model_name]
        
        print("\n" + "-" * 80)
        print("CLASSIFICATION REPORT:")
        print("-" * 80)
        print(classification_report(self.y_test, best_results['y_pred'], 
                                   target_names=['No Disease', 'Disease']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, best_results['y_pred'])
        print("\n" + "-" * 80)
        print("CONFUSION MATRIX:")
        print("-" * 80)
        print(f"True Negatives:  {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives:  {cm[1,1]}")
        
        # Create visualizations
        self._create_evaluation_plots()
        
        return self
    
    def _create_evaluation_plots(self):
        """Create evaluation visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Comparison
        comparison_data = []
        for name, results in self.models.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'F1 Score': results['f1_score'],
                'ROC-AUC': results['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.set_index('Model')[['Accuracy', 'F1 Score', 'ROC-AUC']].plot(
            kind='bar', ax=axes[0, 0], rot=45
        )
        axes[0, 0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Score', fontsize=12)
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].set_ylim([0.7, 1.0])
        
        # 2. Confusion Matrix
        best_results = self.models[self.best_model_name]
        cm = confusion_matrix(self.y_test, best_results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        axes[0, 1].set_title(f'Confusion Matrix - {self.best_model_name}', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('True Label', fontsize=12)
        axes[0, 1].set_xlabel('Predicted Label', fontsize=12)
        
        # 3. ROC Curves
        for name, results in self.models.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC={results["roc_auc"]:.3f})')
        
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[1, 0].set_xlabel('False Positive Rate', fontsize=12)
        axes[1, 0].set_ylabel('True Positive Rate', fontsize=12)
        axes[1, 0].set_title('ROC Curves', fontsize=14, fontweight='bold')
        axes[1, 0].legend(loc='lower right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature Importance (for best model if available)
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            axes[1, 1].barh(range(len(indices)), importances[indices])
            axes[1, 1].set_yticks(range(len(indices)))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 1].set_xlabel('Importance', fontsize=12)
            axes[1, 1].set_title(f'Top 10 Feature Importances - {self.best_model_name}', 
                               fontsize=14, fontweight='bold')
            axes[1, 1].invert_yaxis()
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance not available\nfor this model type',
                          ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Feature Importance', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved visualization: model_evaluation.png")
        plt.show()
    
    def predict_risk(self, patient_data):
        """
        Predict heart disease risk for new patient data.
        
        Parameters:
        -----------
        patient_data : dict
            Dictionary containing patient features
        
        Returns:
        --------
        dict : Prediction results including probability and risk level
        """
        # Convert to dataframe
        patient_df = pd.DataFrame([patient_data])
        
        # Scale features
        patient_scaled = self.scaler.transform(patient_df[self.feature_names])
        
        # Predict
        prediction = self.best_model.predict(patient_scaled)[0]
        probability = self.best_model.predict_proba(patient_scaled)[0]
        
        # Determine risk level
        disease_prob = probability[1]
        if disease_prob < 0.3:
            risk_level = "LOW"
        elif disease_prob < 0.7:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"
        
        return {
            'prediction': 'Disease Detected' if prediction == 1 else 'No Disease',
            'disease_probability': disease_prob,
            'no_disease_probability': probability[0],
            'risk_level': risk_level
        }
    
    def run_pipeline(self):
        """Execute complete ML pipeline."""
        print("\n" + "=" * 80)
        print("HEART DISEASE RISK CLASSIFICATION PIPELINE")
        print("=" * 80)
        
        self.load_data()
        self.explore_data()
        self.preprocess_data()
        self.train_models()
        self.evaluate_models()
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\n✓ Best Model: {self.best_model_name}")
        print(f"✓ ROC-AUC Score: {self.models[self.best_model_name]['roc_auc']:.4f}")
        print(f"✓ Test Accuracy: {self.models[self.best_model_name]['accuracy']:.4f}")
        
        return self


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    # Initialize and run pipeline
    classifier = HeartDiseaseClassifier('heart_disease_uci.csv')
    classifier.run_pipeline()
    
    # Example prediction for a new patient
    print("\n" + "=" * 80)
    print("EXAMPLE: PREDICTING RISK FOR NEW PATIENT")
    print("=" * 80)
    
    sample_patient = {
        'age': 55,
        'sex': 1,  # Male
        'cp': 3,   # Asymptomatic
        'trestbps': 140,
        'chol': 250,
        'fbs': 1,  # Fasting blood sugar > 120 mg/dl
        'restecg': 1,
        'thalch': 150,
        'exang': 1,  # Exercise induced angina
        'oldpeak': 2.0,
        'slope': 2,
        'ca': 2,
        'thal': 2
    }
    
    result = classifier.predict_risk(sample_patient)
    
    print(f"\nPatient Profile:")
    for key, value in sample_patient.items():
        print(f"  {key}: {value}")
    
    print(f"\n{'-'*80}")
    print("PREDICTION RESULTS:")
    print(f"{'-'*80}")
    print(f"Prediction: {result['prediction']}")
    print(f"Disease Probability: {result['disease_probability']:.2%}")
    print(f"No Disease Probability: {result['no_disease_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    
    print("\n" + "=" * 80)
    print("PROJECT COMPLETE")
    print("=" * 80)
    print("\n✓ Model trained and ready for clinical screening")
    print("✓ Visualizations saved to disk")
    print("✓ System can now be used for patient risk assessment")