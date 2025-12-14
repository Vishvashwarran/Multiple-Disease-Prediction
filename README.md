# Multiple Disease Prediction System

This project is a Streamlit-based machine learning application that predicts three diseases:
1. Kidney Disease
2. Liver Disease
3. Parkinson’s Disease

Each disease has its own trained machine learning model, and the app provides predictions along with probability scores.

------------------------------------------------------------

PROJECT OBJECTIVE
The aim of this project is to build a scalable and accurate system for early disease detection. It provides quick predictions based on medical inputs and helps in improved decision-making for healthcare.

------------------------------------------------------------

DATASETS USED
1. Kidney Disease Dataset
2. Indian Liver Patient Dataset
3. Parkinson’s Dataset

Each dataset was cleaned, preprocessed, and trained using separate ML pipelines.

------------------------------------------------------------

MACHINE LEARNING MODELS

Kidney Disease:
- Algorithm: Gradient Boosting Classifier
- Preprocessing: Ordinal Encoding + Standard Scaling using ColumnTransformer
- Outputs: Prediction + Probability score

Liver Disease:
- Algorithm: Random Forest Classifier
- Preprocessing: StandardScaler + OrdinalEncoder + SMOTE for imbalance handling
- Outputs: Prediction + Probability score

Parkinson’s Disease:
- Algorithm: Logistic Regression
- Preprocessing: StandardScaler
- Outputs: Prediction + Probability score

All best models were saved as:
best_pipeline1.pkl
best_pipe2.pkl
best_pipe3.pkl

------------------------------------------------------------

SYSTEM ARCHITECTURE
Frontend: Streamlit UI to collect input and display results  
Backend: Python + Pre-trained ML Models  
Model Layer: Three separate classification models using scikit-learn  

------------------------------------------------------------

FEATURES
- Predicts Kidney, Liver, and Parkinson’s disease
- Probability score displayed with prediction
- Clean UI with sidebar navigation
- Home page displays heatmap visualizations for all datasets
- Easy-to-use form-based input system
- Scalable and extendable architecture

------------------------------------------------------------

HOW TO RUN THE PROJECT

Step 1: Create a virtual environment
python -m venv .venv

Step 2: Activate the environment
Windows:
.venv\Scripts\activate
Mac/Linux:
source .venv/bin/activate

Step 3: Install dependencies
pip install -r requirements.txt

Step 4: Run the Streamlit application
streamlit run app.py

------------------------------------------------------------

PROJECT STRUCTURE

Multiple-Disease-Prediction/
│
├── app.py
├── best_pipeline1.pkl
├── best_pipe2.pkl
├── best_pipe3.pkl
├── kidney_disease.csv
├── indian_liver_patient.csv
├── parkinsons.csv
├── README.md
└── requirements.txt

------------------------------------------------------------

REQUIREMENTS.TXT (Example)

streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
xgboost
imblearn

------------------------------------------------------------

EVALUATION METRICS USED

Accuracy  
Precision, Recall, F1-score  
Confusion Matrix  
ROC-AUC (for supported models)  
Heatmap Correlation Analysis  

------------------------------------------------------------

PROJECT DELIVERABLES

- Source code for model training and deployment
- Streamlit application (local deployment)
- Documentation of methodologies
- Prediction interface with probability display
- Dataset heatmaps for insights

------------------------------------------------------------

