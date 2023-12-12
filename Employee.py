import streamlit as st
import pandas as pd
import io
import zipfile
from operator import index
import streamlit as st
import plotly.express as px
import ts as ts
from Cython import inline
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as tck
# import modules
import pandas  # for dataframes
import matplotlib.pyplot as plt  # for plotting graphs
import seaborn as sns  # for plotting graphs
from sklearn.cluster import KMeans
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn import preprocessing

with st.sidebar:
    st.image("employee.gif")
    st.title("Employee Churn")
    choice = st.radio("Navigation",
                      ["Profiling", "Stayed vs. Left: Employee Data Comparison",
                    "Descriptive Statistics Overview",
                       "Employees Left", "Show Value Counts", "Number of Projects Distribution",
                       "Time Spent in Company",
                       "Employee Count by Features", "Clustering of Employees who Left",
                       "Employee Clustering Analysis"])
    st.info("Employee Churn App provides a user-friendly interface for HR professionals and data enthusiasts to "
            "explore and gain insights from employee data, with a focus on predicting and understanding employee "
            "turnover.")

if choice == "Profiling":
    st.title("Data Profiling Dashboard")
    # Assuming your CSV file is in the same directory as the script
    csv_path = "HR_comma_sep.csv"

    # Check if the file exists
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        a = df.head()
        st.dataframe(a)

if choice == "Stayed vs. Left: Employee Data Comparison":
    st.title("Employee Retention Analysis: Comparing Characteristics of Stayed and Left Groups")
    # Assuming your CSV file is in the same directory as the script
    csv_path = "HR_comma_sep.csv"

    # Check if the file exists
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        left = df.groupby('left')
        b = left.mean()
        st.dataframe(b)

    else:
        st.error("CSV file not found. Please check the file path.")

if choice == "Descriptive Statistics Overview":
    st.title(" Employee Attrition Analysis")
    # Assuming your CSV file is in the same directory as the script
    csv_path = "HR_comma_sep.csv"

    # Check if the file exists
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        b = df.describe()
        st.dataframe(b)

    else:
        st.error("CSV file not found. Please check the file path.")

if choice == "Employees Left":
    st.title("Data Visualization")
    # Assuming your CSV file is in the same directory as the script
    csv_path = "HR_comma_sep.csv"

    # Check if the file exists
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        left_count = df.groupby('left').count()
        st.bar_chart(left_count['satisfaction_level'])

    else:
        st.error("CSV file not found. Please check the file path.")

if choice == "Show Value Counts":
    st.title("Employee Left Counts")
    csv_path = "HR_comma_sep.csv"
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        left_counts = df.left.value_counts()

        # Display the value counts using Streamlit
        st.write(left_counts)

        # Optionally, you can create a bar chart as well
        st.bar_chart(left_counts)

if choice == "Number of Projects Distribution":
    st.title("Employees' Project Distribution")

    # Assuming your CSV file is in the same directory as the script
    csv_path = "HR_comma_sep.csv"

    # Check if the file exists
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)

        # Plotting Number of Projects Analysis
        st.subheader('Number of Projects Analysis')
        num_projects = df.groupby('number_project').count()
        plt.bar(num_projects.index.values, num_projects['satisfaction_level'])
        plt.xlabel('Number of Projects')
        plt.ylabel('Number of Employees')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Show the plot in Streamlit
        st.pyplot()

if choice == "Time Spent in Company":
    st.title("Data Visualization")
    # Assuming your CSV file is in the same directory as the script
    csv_path = "HR_comma_sep.csv"

    # Check if the file exists
    if os.path.isfile(csv_path):
        data = pd.read_csv(csv_path)
        time_spent = data.groupby('time_spend_company').count()
        plt.bar(time_spent.index.values, time_spent['satisfaction_level'])
        plt.xlabel('Number of Years Spent in Company')
        plt.ylabel('Number of Employees')

        # Display the plot in Streamlit
        st.pyplot()

if choice == "Employee Count by Features":
    csv_path = "HR_comma_sep.csv"

    # Check if the file exists
    if os.path.isfile(csv_path):
        data = pd.read_csv(csv_path)
        features = ['number_project', 'time_spend_company', 'Work_accident', 'left', 'promotion_last_5years',
                    'Departments ', 'salary']

        st.title("Data Visualization")

        # Create subplots
        fig, axes = plt.subplots(4, 2, figsize=(10, 15))

        for i, j in enumerate(features):
            row, col = divmod(i, 2)
            sns.countplot(x=j, data=data, ax=axes[row, col])
            axes[row, col].tick_params(axis="x", rotation=90)
            axes[row, col].set_title(f"No. of Employees - {j}")

        # Adjust layout
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

if choice == "Clustering of Employees who Left":
    csv_path = "HR_comma_sep.csv"

    # Check if the file exists
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        # Label Encoding for 'salary' and 'Departments ' columns
        le = preprocessing.LabelEncoder()
        df['salary'] = le.fit_transform(df['salary'])
        df['Departments '] = le.fit_transform(df['Departments '])

        # Splitting data into Feature and Target
        X = df[['satisfaction_level', 'last_evaluation', 'number_project',
                'average_montly_hours', 'time_spend_company', 'Work_accident',
                'promotion_last_5years', 'Departments ', 'salary']]
        y = df['left']

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Create Gradient Boosting Classifier
        gb = GradientBoostingClassifier()

        # Train the model using the training sets
        gb.fit(X_train, y_train)

        # Predict the response for the test dataset
        y_pred = gb.predict(X_test)

        # Model Evaluation Metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)

        # Display Metrics in Streamlit
        st.title("Gradient Boosting Classifier Model Evaluation")
        st.write("Accuracy:", accuracy)
        st.write("Precision:", precision)
        st.write("Recall:", recall)

        y_pred_all = gb.predict(X)

        # Create a DataFrame for actual vs predicted values for all data
        diff_all_df = pd.DataFrame({
            'Sample': range(len(y)),
            'Actual': y,
            'Predicted': y_pred_all
        })

        # Count correct and incorrect predictions
        diff_all_df['Correct'] = (diff_all_df['Actual'] == diff_all_df['Predicted']).astype(int)
        diff_counts = diff_all_df.groupby('Correct').size().reset_index(name='Count')

        # Stacked bar chart for the difference between actual and predicted values for all data
        fig_diff_all = px.bar(diff_counts, x='Correct', y='Count', color='Correct',
                              labels={'Correct': 'Prediction Correctness', 'Count': 'Number of Samples'},
                              title='Actual vs Predicted for All Data',
                              color_discrete_map={0: 'red', 1: 'green'})

        # Layout adjustments
        fig_diff_all.update_layout(showlegend=False)

        # Show the plot in Streamlit
        st.plotly_chart(fig_diff_all)

if choice == "Employee Clustering Analysis":
    st.title("Employee Clustering Analysis")

    csv_path = "HR_comma_sep.csv"
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)

        le = preprocessing.LabelEncoder()
        df['salary'] = le.fit_transform(df['salary'])
        df['Departments '] = le.fit_transform(df['Departments '])

        # Features for clustering
        features_for_clustering = ['satisfaction_level', 'last_evaluation', 'number_project',
                                   'average_montly_hours', 'time_spend_company', 'Work_accident',
                                   'promotion_last_5years', 'Departments ', 'salary']

        X = df[features_for_clustering]

        # Perform KMeans clustering
        num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        # Visualize the clusters
        fig_clusters = px.scatter_3d(df, x='satisfaction_level', y='last_evaluation', z='average_montly_hours',
                                     color='Cluster', opacity=0.7, title='Employee Clusters')

        # Show the plot in Streamlit
        st.plotly_chart(fig_clusters)
