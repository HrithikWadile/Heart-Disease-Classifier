# Heart-Disease-Classifier❤️ Heart Disease Risk Classification Using Machine Learning

A complete end-to-end machine learning pipeline that predicts heart disease risk using the UCI Heart Disease dataset.
The project performs data exploration, preprocessing, model training, evaluation, visualization, and risk prediction for new patient inputs.

📌 Project Overview
This project implements a comprehensive diagnostic ML system to classify whether a patient is likely to have heart disease based on clinical features such as:
Age
Sex
Chest pain type
Cholesterol levels
Resting blood pressure
Fasting blood sugar
Exercise-induced angina
Thalassemia
Major vessels
ST depression
and more…

The pipeline automatically:
✔ Loads & inspects dataset
✔ Performs exploratory data analysis (EDA)
✔ Handles missing values & preprocessing
✔ Encodes categorical features
✔ Splits & scales data
✔ Trains multiple ML models
✔ Evaluates performance using accuracy, F1-score & ROC-AUC
✔ Visualizes results
✔ Predicts risk for any new patient

🚀 Features
🔎 1. Data Exploration & Visualization
Missing value analysis
Target variable distribution
Summary statistics
Age distribution by disease
Chest pain & sex distribution plots
Correlation heatmap
Automatically saves plots:
heart_disease_eda.png
correlation_heatmap.png

⚙️ 2. Data Preprocessing
Binary target creation (0 = No Disease, 1 = Disease)
Label encoding for categorical features
Handling missing numerical values (median imputation)
Train-test split using stratification
Feature scaling using StandardScaler

🤖 3. Machine Learning Models Used
The pipeline trains and evaluates 4 different models:
Model	Strength
Logistic Regression	Interpretable baseline model
Random Forest Classifier	Strong, non-linear, feature importance
Gradient Boosting Classifier	High performance boosting
Support Vector Classifier (SVM)	Effective in complex boundaries
Each model is evaluated using:
Cross-validation accuracy
Test accuracy
F1 score
ROC-AUC score
The best model is automatically selected based on ROC-AUC.

📊 4. Model Evaluation Visualizations
The system generates:
Model comparison bar charts
Confusion matrix heatmap
ROC curves for all models
Top 10 feature importances (if supported)
Plots saved as:
model_evaluation.png

🧪 5. Risk Prediction for New Patients
The system accepts a dictionary of patient clinical values and returns:
Predicted class
Disease probability
Healthy probability
Risk category: LOW / MODERATE / HIGH

Example:
result = classifier.predict_risk(sample_patient)
📁 Project Structure
│── app.py / notebook.py
│── heart_disease_uci.csv
│── heart_disease_eda.png
│── correlation_heatmap.png
│── model_evaluation.png
│── README.md

🧠 How the Pipeline Works
1️⃣ Load dataset
2️⃣ Perform EDA
3️⃣ Clean and preprocess
4️⃣ Train ML models
5️⃣ Compare performance
6️⃣ Select best model
7️⃣ Evaluate with visualizations
8️⃣ Predict heart disease for new patient

Run with:
python app.py

📈 Model Results (Example)
The project prints:
Best model
ROC-AUC score
Test accuracy
F1-score
Confusion matrix
Classification report

📥 Dataset
This project uses the UCI Heart Disease Dataset, containing 13 clinical features and a target variable.

🧑‍⚕️ Use Cases
Healthcare analytics
Clinical decision support systems
Research and academic projects
Predictive modeling for hospitals

🛠️ Technologies Used
Python
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
Machine Learning Algorithms

🏁 Conclusion
This project provides a robust ML-based diagnostic system capable of screening heart disease risk with multiple model comparison, visual insights, and real-time prediction capability.

Screenshots:
<img width="793" height="637" alt="image" src="https://github.com/user-attachments/assets/47b36e95-554f-46fc-a6b8-98b76062396d" />
<img width="700" height="635" alt="image" src="https://github.com/user-attachments/assets/413d389a-4979-4a93-b906-62760a101822" />
<img width="797" height="635" alt="image" src="https://github.com/user-attachments/assets/f845f754-f94a-4117-9cf8-fc72b0d0c52f" />

