# Student Performance Prediction

## Overview
This project analyzes the actual student performance dataset to identify academic trends, measure key metrics, and predict final grades using machine learning. It is designed as a professional data analytics project for academic and educational decision support.

## Project Objective
The objective is to inspect the real dataset, clean and preprocess it, understand relationships between academic and demographic variables, calculate KPIs, compare regression models, and provide an interactive dashboard for analysis and prediction.

## Problem Statement
Educational institutions need a systematic way to understand patterns in student performance and detect students who may need additional support. A data-driven approach can help highlight associations between attendance, study behavior, prior scores, and outcomes without claiming direct causation.

## Dataset
The project uses the real CSV file available in the workspace: `student_performance_updated_1000.csv`. The dataset includes academic metrics and demographic information for a student population.

## Dataset Features
The dataset contains variables including:
- StudentID
- Name
- Gender
- AttendanceRate
- StudyHoursPerWeek
- PreviousGrade
- ExtracurricularActivities
- ParentalSupport
- FinalGrade
- Study Hours
- Attendance (%)
- Online Classes Taken

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Streamlit
- Joblib
- python-docx

## Project Workflow
1. Dataset inspection and automatic target detection
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. KPI and statistical analysis
5. Feature engineering
6. Machine learning model training
7. Model evaluation and comparison
8. Prediction interface and dashboard

## EDA
The exploratory analysis focuses on attendance patterns, study time, prior achievement, gender patterns, and support-related factors using actual dataset values. This helps identify which variables show meaningful associations with performance.

## KPIs
The project calculates KPIs directly from the dataset, including:
- Total Students
- Average Score
- Average Attendance
- Pass Rate
- High Performer Percentage
- At-Risk Percentage
- Median Score

## Machine Learning Models
The target variable is numeric, so the project applies regression models. The pipeline includes preprocessing and compares multiple models for predictive performance.

## Model Evaluation
Training and evaluation use proper train-test splitting with a fixed random state. The model comparison is based on:
- Mean Absolute Error
- Root Mean Squared Error
- R-squared (R2)

## Streamlit Dashboard
The dashboard provides a clean and interactive interface with pages for:
- Home
- Data overview
- EDA
- KPI dashboard
- Correlation analysis
- Machine learning overview
- Student prediction
- About project

## Key Insights
Key insights are calculated from the dataset and are reported as observed patterns. Importantly, the project does not claim causal effects from correlation alone.

## Project Structure
```text
Student_Performance_Prediction/
├── app.py
├── analysis.py
├── requirements.txt
├── README.md
├── Student_Performance_Project_Report.docx
├── student_performance.csv
├── data/
│   └── student_performance.csv
├── models/
│   ├── model.pkl
├── visualizations/
│   └── generated charts
├── notebooks/
│   └── student_performance_analysis.ipynb
└── .gitignore
```

## Installation
```bash
pip install -r requirements.txt
```

## Run Streamlit
```bash
streamlit run app.py
```

## Requirements
- Python 3.10+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- scikit-learn
- Streamlit
- Joblib

## Future Enhancements
- Deep learning models
- Explainable AI
- Real-time analytics
- Large-scale institutional data integration
- Cloud deployment

## Author
Kalyan Mahata
