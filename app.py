from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / 'data'
MODELS_DIR = PROJECT_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title='Student Performance Prediction', layout='wide')


def detect_dataset_path():
    candidates = [
        PROJECT_DIR / 'student_performance.csv',
        DATA_DIR / 'student_performance.csv',
        PROJECT_DIR.parent / 'student_performance_updated_1000.csv',
        PROJECT_DIR.parent / 'student_performance.csv',
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_target_column(df):
    priority_order = [
        'finalgrade', 'final grade', 'finalscore', 'final score', 'performance', 'performancecategory',
        'grade', 'gpa', 'score', 'marks', 'examscore', 'exam score'
    ]
    normalized_columns = {str(col).strip().lower().replace(' ', ''): col for col in df.columns if str(col).lower() not in {'studentid', 'name'}}
    for candidate in priority_order:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    for col in df.columns:
        normalized = str(col).strip().lower().replace(' ', '')
        if 'grade' in normalized or 'score' in normalized or 'performance' in normalized:
            if col.lower() not in {'studentid', 'name'}:
                return col
    return None


def load_and_prepare_data():
    path = detect_dataset_path()
    if path is None:
        st.error('Dataset not found. Please place the student performance CSV in the project folder or the data folder.')
        st.stop()

    df = pd.read_csv(path)
    if df.empty:
        st.error('The dataset is empty. Please provide a valid student performance dataset.')
        st.stop()

    data = df.copy()
    data = data.drop_duplicates().reset_index(drop=True)

    for col in ['StudentID', 'Name']:
        if col in data.columns:
            data = data.drop(columns=[col])

    for col in ['Gender', 'ParentalSupport']:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip().str.title().replace({'Nan': 'Unknown', 'None': 'Unknown'})

    if 'Online Classes Taken' in data.columns:
        cleaned = data['Online Classes Taken'].astype(str).str.strip().str.lower()
        mapping = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0}
        data['Online Classes Taken'] = cleaned.map(mapping).fillna(pd.to_numeric(cleaned, errors='coerce'))
        data['Online Classes Taken'] = pd.to_numeric(data['Online Classes Taken'], errors='coerce')

    target = find_target_column(data)
    if target is None:
        st.error('Could not identify a valid target variable automatically. Please inspect the dataset columns and choose a grade/performance target.')
        st.stop()

    data[target] = pd.to_numeric(data[target], errors='coerce')
    data = data.dropna(subset=[target]).reset_index(drop=True)
    data['PerformanceCategory'] = pd.cut(
        data[target],
        bins=[-1, 59, 74, 84, 101],
        labels=['At Risk', 'Average', 'High Performing', 'Excellent'],
        right=False,
    )
    return data, target


def build_model(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_cols),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols),
    ])

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42, n_estimators=300),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
    }

    results = []
    for name, estimator in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', estimator),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        results.append({
            'Model': name,
            'MAE': mean_absolute_error(y_test, pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
            'R2': r2_score(y_test, pred),
            'Pipeline': pipeline,
        })

    best = min(results, key=lambda item: item['MAE'])
    joblib.dump(best['Pipeline'], MODELS_DIR / 'model.pkl')
    return best, results


def ensure_model(df, target):
    model_path = MODELS_DIR / 'model.pkl'
    if not model_path.exists():
        best, _ = build_model(df, target)
        return best
    return joblib.load(model_path)


def sidebar_filters(df):
    st.sidebar.header('Dataset Filters')
    gender_values = ['All'] + sorted(df['Gender'].dropna().unique().tolist()) if 'Gender' in df.columns else ['All']
    gender = st.sidebar.selectbox('Gender', gender_values)

    attendance_values = [
        'All',
    ]
    if 'AttendanceRate' in df.columns:
        attendance_values = ['All']
        attendance_min = float(df['AttendanceRate'].min())
        attendance_max = float(df['AttendanceRate'].max())
        attendance_range = st.sidebar.slider('Attendance range', attendance_min, attendance_max, (attendance_min, attendance_max))
    else:
        attendance_range = (0, 100)

    score_min = float(df['FinalGrade'].min()) if 'FinalGrade' in df.columns else float(df.iloc[:, -1].min())
    score_max = float(df['FinalGrade'].max()) if 'FinalGrade' in df.columns else float(df.iloc[:, -1].max())
    score_range = st.sidebar.slider('Score range', score_min, score_max, (score_min, score_max))

    return {'gender': gender, 'attendance_range': attendance_range, 'score_range': score_range}


def filter_data(df, filters, target):
    data = df.copy()
    if 'Gender' in data.columns and filters['gender'] != 'All':
        data = data[data['Gender'] == filters['gender']]

    if 'AttendanceRate' in data.columns:
        low, high = filters['attendance_range']
        data = data[(data['AttendanceRate'] >= low) & (data['AttendanceRate'] <= high)]

    low, high = filters['score_range']
    data = data[(data[target] >= low) & (data[target] <= high)]
    return data


def kpi_cards(df, target):
    attendance_col = 'AttendanceRate' if 'AttendanceRate' in df.columns else 'Attendance (%)'
    return {
        'Total Students': len(df),
        'Average Score': float(df[target].mean()),
        'Average Attendance': float(df[attendance_col].mean()),
        'Pass Rate (%)': float((df[target] >= 60).mean() * 100),
        'High Performer (%)': float((df[target] >= 85).mean() * 100),
        'At-Risk (%)': float((df[target] < 60).mean() * 100),
    }


def home_page():
    st.title('STUDENT PERFORMANCE PREDICTION')
    st.caption('Data Analytics and Machine Learning Based Student Performance Analysis')
    st.markdown('''
    This project analyzes real student data to identify performance patterns, highlight academic risk, and support informed educational decisions.
    - Objective: discover which academic and demographic factors are associated with final outcomes.
    - Dataset: actual student academic records with attendance, study time, and support indicators.
    - Methods: data cleaning, EDA, KPI analysis, model training, and prediction.
    - Tools: Python, Pandas, Plotly, Scikit-learn, Streamlit.
    ''')

    st.subheader('Project Summary')
    st.info('The target variable is determined automatically from the dataset and the project uses the actual data values rather than fabricated statistics.')


def overview_page(df):
    st.header('Data Overview')
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader('Dataset Summary')
    st.dataframe(df.describe(include='all').transpose(), use_container_width=True)


def eda_page(df, target):
    st.header('EDA')

    if 'Gender' in df.columns:
        gender_avg = df.groupby('Gender')[target].mean().reset_index()
        fig = px.bar(gender_avg, x='Gender', y=target, color='Gender', title='Average Final Grade by Gender')
        st.plotly_chart(fig, use_container_width=True)

    if 'AttendanceRate' in df.columns:
        fig2 = px.scatter(df, x='AttendanceRate', y=target, color='Gender' if 'Gender' in df.columns else None, title='Attendance vs Final Grade')
        st.plotly_chart(fig2, use_container_width=True)

    if 'StudyHoursPerWeek' in df.columns:
        fig3 = px.scatter(df, x='StudyHoursPerWeek', y=target, title='Study Hours vs Final Grade')
        st.plotly_chart(fig3, use_container_width=True)

    if 'PerformanceCategory' in df.columns:
        fig4 = px.pie(df['PerformanceCategory'].value_counts().reset_index(), names='PerformanceCategory', values='count', title='Performance Category Distribution')
        st.plotly_chart(fig4, use_container_width=True)


def dashboard_page(df, target):
    st.header('KPI Dashboard')
    metrics = kpi_cards(df, target)
    cols = st.columns(4)
    cols[0].metric('Total Students', f"{metrics['Total Students']:,}")
    cols[1].metric('Average Score', f"{metrics['Average Score']:.2f}")
    cols[2].metric('Average Attendance', f"{metrics['Average Attendance']:.2f}%")
    cols[3].metric('Pass Rate', f"{metrics['Pass Rate (%)']:.2f}%")

    cols2 = st.columns(3)
    cols2[0].metric('High Performer %', f"{metrics['High Performer (%)']:.2f}%")
    cols2[1].metric('At-Risk %', f"{metrics['At-Risk (%)']:.2f}%")
    cols2[2].metric('Median Score', f"{df[target].median():.2f}")


def correlation_page(df):
    st.header('Correlation Analysis')
    numeric_data = df.select_dtypes(include=[np.number])
    corr = numeric_data.corr().round(3)
    fig = px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale='RdBu_r', title='Correlation Heatmap')
    st.plotly_chart(fig, use_container_width=True)


def model_page(df, target):
    st.header('Machine Learning')
    model_path = MODELS_DIR / 'model.pkl'
    if not model_path.exists():
        model, results = build_model(df, target)
        st.success('Model trained successfully on the actual dataset.')
    else:
        model = joblib.load(model_path)
        st.success(f'Model loaded from {model_path}')

    st.write('Problem type: Regression because the target variable is a continuous numeric grade value.')
    st.write('Selected models: Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor.')


def prediction_page(df, target):
    st.header('Student Prediction Interface')
    model_path = MODELS_DIR / 'model.pkl'
    if not model_path.exists():
        model = ensure_model(df, target)
    else:
        model = joblib.load(model_path)

    st.subheader('Enter student details')
    with st.form('student_prediction'):
        gender = st.selectbox('Gender', sorted(df['Gender'].dropna().unique().tolist()))
        attendance = st.slider('Attendance Rate (%)', 0, 100, 85)
        study_hours = st.slider('Study Hours Per Week', 0, 40, 15)
        previous_grade = st.slider('Previous Grade', 0, 100, 75)
        extracurricular = st.slider('Extracurricular Activities', 0, 5, 1)
        parental_support = st.selectbox('Parental Support', sorted(df['ParentalSupport'].dropna().unique().tolist()))
        online_classes = st.selectbox('Online Classes Taken', [0, 1])
        study_hours_raw = st.slider('Study Hours (hours per day)', 0.0, 10.0, 3.5, step=0.1)
        attendance_percent = st.slider('Attendance (%)', 0, 100, 80)
        submitted = st.form_submit_button('Predict Student Performance')

    if submitted:
        sample = pd.DataFrame([
            {
                'Gender': gender,
                'AttendanceRate': attendance,
                'StudyHoursPerWeek': study_hours,
                'PreviousGrade': previous_grade,
                'ExtracurricularActivities': extracurricular,
                'ParentalSupport': parental_support,
                'FinalGrade': previous_grade,
                'Study Hours': study_hours_raw,
                'Attendance (%)': attendance_percent,
                'Online Classes Taken': online_classes,
            }
        ])
        prediction = model.predict(sample)[0]
        st.metric('Predicted Final Grade', f'{prediction:.2f}')
        if prediction >= 85:
            st.success('High Performing')
        elif prediction >= 60:
            st.info('Average / Passing')
        else:
            st.warning('At Risk')


def about_page():
    st.header('About Project')
    st.markdown('''
    This project is designed for academic analytics and educational decision support. It helps identify patterns in student performance, monitor academic risk, and use a trained model to estimate likely outcomes based on the available historical data.
    Predictions should support human judgment rather than replace it.
    ''')


def main():
    df, target = load_and_prepare_data()
    filters = sidebar_filters(df)
    filtered = filter_data(df, filters, target)

    nav = st.sidebar.radio('Navigation', ['HOME', 'DATA OVERVIEW', 'EDA', 'KPI DASHBOARD', 'CORRELATION ANALYSIS', 'MACHINE LEARNING', 'PREDICTION', 'ABOUT PROJECT'])

    if nav == 'HOME':
        home_page()
    elif nav == 'DATA OVERVIEW':
        overview_page(filtered)
    elif nav == 'EDA':
        eda_page(filtered, target)
    elif nav == 'KPI DASHBOARD':
        dashboard_page(filtered, target)
    elif nav == 'CORRELATION ANALYSIS':
        correlation_page(filtered)
    elif nav == 'MACHINE LEARNING':
        model_page(filtered, target)
    elif nav == 'PREDICTION':
        prediction_page(filtered, target)
    elif nav == 'ABOUT PROJECT':
        about_page()


if __name__ == '__main__':
    main()
