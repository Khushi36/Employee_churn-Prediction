 import streamlit as st
        import pandas as pd
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.cluster import KMeans
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn import metrics
        import plotly.express as px

        # Load the pre-trained model
        model = pickle.load(open('pretrained models/employee_churn_model.pkl', 'rb'))

        # Load the CSV file
        csv_path = "HR_comma_sep.csv"
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
        else:
            st.error("CSV file not found. Please check the file path.")

        features_for_prediction = ['satisfaction_level', 'last_evaluation', 'number_project',
                                   'average_montly_hours', 'time_spend_company', 'Work_accident',
                                   'promotion_last_5years', 'Departments ', 'salary']


        # Function to handle user input for clustering analysis
        def user_report():
            # Assuming df is your DataFrame containing historical employee data
            csv_path = "HR_comma_sep.csv"
            df = pd.read_csv(csv_path)

            # Create an empty dictionary to store user input data
            user_data = {}

            # Iterate over features to get user input
            for feature in features_for_prediction:
                if df[feature].dtype == 'float64':
                    # If the feature is of type float, use float values for slider
                    user_data[feature] = st.sidebar.slider(f'Select {feature}', float(df[feature].min()),
                                                           float(df[feature].max()), float(df[feature].mean()))
                elif df[feature].dtype == 'int64':
                    # If the feature is of type int, use int values for slider
                    user_data[feature] = st.sidebar.slider(f'Select {feature}', int(df[feature].min()),
                                                           int(df[feature].max()), int(df[feature].mean()))
                else:
                    # Handle other data types as needed
                    user_data[feature] = st.sidebar.text_input(f'Enter {feature}', df[feature].iloc[0])

            return pd.DataFrame(user_data, index=[0])


        def descriptive_statistics():
            st.title(" Employee Attrition Analysis")
            b = df.describe()
            st.dataframe(b)

        # Apply label encoding to categorical columns
        df['Departments '] = LabelEncoder().fit_transform(df['Departments '])
        df['salary'] = LabelEncoder().fit_transform(df['salary'])

        # Handle missing values if any
        if df.isnull().any().any():
            df = df.fillna(df.mean())

        # Streamlit App
        with st.sidebar:
            st.image("images/employee.gif")
            st.title("Employee Churn")
            choice = st.radio("Navigation",
                              ["Profiling", "Stayed vs. Left: Employee Data Comparison",
                               "Descriptive Statistics Overview",
                               "Employees Left", "Show Value Counts", "Number of Projects Distribution",
                               "Time Spent in Company",
                               "Employee Count by Features", "Clustering of Employees who Left",
                               "Employee Clustering Analysis", "Predict Churn"])
            st.info(
                "Employee Churn App provides a user-friendly interface for HR professionals and data enthusiasts to "
                "explore and gain insights from employee data, with a focus on predicting and understanding employee "
                "turnover.")

        if choice == "Profiling":
            st.title("Data Profiling Dashboard")
            a = df.head()
            st.dataframe(a)

        if choice == "Stayed vs. Left: Employee Data Comparison":
            st.title("Employee Retention Analysis: Comparing Characteristics of Stayed and Left Groups")
            left = df.groupby('left')
            b = left.mean()
            st.dataframe(b)

        if choice == "Descriptive Statistics Overview":
            descriptive_statistics()

        if choice == "Employees Left":
            st.title("Data Visualization")
            left_count = df.groupby('left').count()
            st.bar_chart(left_count['satisfaction_level'])

        if choice == "Show Value Counts":
            st.title("Employee Left Counts")
            left_counts = df.left.value_counts()
            st.write(left_counts)
            st.bar_chart(left_counts)

        if choice == "Number of Projects Distribution":
            st.title("Employees' Project Distribution")
            num_projects = df.groupby('number_project').count()
            plt.bar(num_projects.index.values, num_projects['satisfaction_level'])
            plt.xlabel('Number of Projects')
            plt.ylabel('Number of Employees')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

        if choice == "Time Spent in Company":
            st.title("Data Visualization")
            time_spent = df.groupby('time_spend_company').count()
            plt.bar(time_spent.index.values, time_spent['satisfaction_level'])
            plt.xlabel('Number of Years Spent in Company')
            plt.ylabel('Number of Employees')
            st.pyplot()

        if choice == "Employee Count by Features":
            st.title("Data Visualization")
            features = ['number_project', 'time_spend_company', 'Work_accident', 'left', 'promotion_last_5years',
                        'Departments ', 'salary']

            fig, axes = plt.subplots(4, 2, figsize=(10, 15))

            for i, j in enumerate(features):
                row, col = divmod(i, 2)
                sns.countplot(x=j, data=df, ax=axes[row, col])
                axes[row, col].tick_params(axis="x", rotation=90)
                axes[row, col].set_title(f"No. of Employees - {j}")

            plt.tight_layout()
            st.pyplot(fig)

        if choice == "Clustering of Employees who Left":
            X = df[['satisfaction_level', 'last_evaluation', 'number_project',
                    'average_montly_hours', 'time_spend_company', 'Work_accident',
                    'promotion_last_5years', 'Departments ', 'salary']]
            y = df['left']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            gb = GradientBoostingClassifier()
            gb.fit(X_train, y_train)
            y_pred = gb.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            st.title("Gradient Boosting Classifier Model Evaluation")
            st.write("Accuracy:", accuracy)
            st.write("Precision:", precision)
            st.write("Recall:", recall)

            y_pred_all = gb.predict(X)
            diff_all_df = pd.DataFrame({
                'Sample': range(len(y)),
                'Actual': y,
                'Predicted': y_pred_all
            })
            diff_all_df['Correct'] = (diff_all_df['Actual'] == diff_all_df['Predicted']).astype(int)
            diff_counts = diff_all_df.groupby('Correct').size().reset_index(name='Count')
            fig_diff_all = px.bar(diff_counts, x='Correct', y='Count', color='Correct',
                                  labels={'Correct': 'Prediction Correctness', 'Count': 'Number of Samples'},
                                  title='Actual vs Predicted for All Data',
                                  color_discrete_map={0: 'red', 1: 'green'})
            fig_diff_all.update_layout(showlegend=False)
            st.plotly_chart(fig_diff_all)

        if choice == "Employee Clustering Analysis":
            st.title("Employee Clustering Analysis")
            user_data = user_report()
            user_data['Departments '] = LabelEncoder().fit_transform(user_data['Departments '])
            user_data['salary'] = LabelEncoder().fit_transform(user_data['salary'])
            if user_data.isnull().any().any():
                user_data = user_data.fillna(user_data.mean())
            features_for_clustering = ['satisfaction_level', 'last_evaluation', 'number_project',
                                       'average_montly_hours', 'time_spend_company', 'Work_accident',
                                       'promotion_last_5years', 'Departments ', 'salary']
            scaler = StandardScaler()
            user_data[['satisfaction_level', 'last_evaluation', 'average_montly_hours']] = scaler.fit_transform(
                user_data[['satisfaction_level', 'last_evaluation', 'average_montly_hours']])
            X = df[features_for_clustering]
            num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X)
            fig_clusters = px.scatter_3d(df, x='satisfaction_level', y='last_evaluation', z='average_montly_hours',
                                         color='Cluster', opacity=0.7, title='Employee Clusters')
            st.plotly_chart(fig_clusters)

        if choice == "Predict Churn":
            st.title("Employee Churn Prediction")
            user_report_data = {
                'satisfaction_level': st.sidebar.slider('Satisfaction Level', 0.0, 1.0, 0.5),
                'last_evaluation': st.sidebar.slider('Last Evaluation', 0.0, 1.0, 0.5),
                'number_project': st.sidebar.slider('Number of Projects', 2, 7, 4),
                'average_montly_hours': st.sidebar.slider('Average Monthly Hours', 80, 300, 160),
                'time_spend_company': st.sidebar.slider('Time Spent in Company', 2, 10, 3),
                'Work_accident': st.sidebar.selectbox('Work Accident', [0, 1]),
                'promotion_last_5years': st.sidebar.selectbox('Promotion in Last 5 Years', [0, 1]),
                'Departments ': st.sidebar.selectbox('Department', df['Departments '].unique()),
                'salary': st.sidebar.selectbox('Salary', df['salary'].unique())
            }
            user_data = pd.DataFrame(user_report_data, index=[0])
            st.header('Employee Data for Prediction')
            st.write(user_data)
            features_for_prediction = ['satisfaction_level', 'last_evaluation', 'number_project',
                                       'average_montly_hours', 'time_spend_company', 'Work_accident',
                                       'promotion_last_5years', 'Departments ', 'salary']
            missing_columns = set(features_for_prediction) - set(user_data.columns)
            if missing_columns:
                st.error(f"Columns {missing_columns} not found in user data.")
            else:
                X_pred = user_data[features_for_prediction]
                X_pred.columns = ['satisfaction_level', 'last_evaluation', 'number_project',
                                  'average_montly_hours', 'time_spend_company', 'Work_accident',
                                  'promotion_last_5years', 'Departments ', 'salary']
                churn_prediction = model.predict(X_pred)
                st.subheader('Churn Prediction Result')
                st.write(churn_prediction)
