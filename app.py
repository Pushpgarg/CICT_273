from regex import F
import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from ast import literal_eval

# -------------------------------------------------
# PAGE CONFIG: FULL-WINDOW LAYOUT & TITLE
# -------------------------------------------------
st.set_page_config(page_title="AI Project Dashboard", layout="wide")

# -------------------------------------------------
# CUSTOM CSS STYLING
# -------------------------------------------------
st.markdown("""
<style>
/* Force a light background (some dark theme elements might still override this) */
body {
    background-color: #ffffff;
    color: #333;
    font-family: 'Helvetica', sans-serif;
    margin: 0;
    padding: 0;
}
header, .reportview-container {
    padding: 0;
    margin: 0;
}
.stButton>button {
    background-color: #1976d2;
    color: white;
    border: none;
    padding: 10px 24px;
    font-size: 16px;
    border-radius: 8px;
}
.sidebar .sidebar-content {
    background-color: #f8f9fa;
}
.container {
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
page = st.sidebar.radio("Navigate", ["Home", "Explore", "Custom Model"])

# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
def home_page():
    st.title("AI Project Dashboard 🤖")
    st.subheader("Abstract & Project Description")
    st.write("""
    Welcome to our interactive AI project dashboard!  
    Here, we experiment with a Random Forest model on a given dataset, tweaking hyperparameters and feature selections to optimize performance.  
    Use the sidebar to explore the experiment results or run your own custom model. Enjoy the interactivity and have fun! 😊
    """)
    # Example banner image
    st.image("Poster.png" , width=700)

# -------------------------------------------------
# EXPLORE PAGE
# -------------------------------------------------
def explore_page():
    st.title("Explore Experiment Results 📊")
    
    # Use full window columns
    left_col, right_col = st.columns([1, 3])
    
    # Left panel for dynamic filtering options
    with left_col:
        st.header("Filter Options")
        # Test accuracy range
        test_acc_range = st.slider("Select Test Accuracy Range", 0.0, 1.0, (0.5, 1.0), 0.01)
        # Train accuracy range
        train_acc_range = st.slider("Select Train Accuracy Range", 0.0, 1.0, (0.5, 1.0), 0.01)
        
        # Filtering by selected features from the dataset
        available_features = [
            'age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal'
        ]
        chosen_features = st.multiselect("Filter rows by selected feature(s):", available_features)
    
    # Right panel for displaying filtered results and plots
    with right_col:
        st.subheader("Experiment Results Table & Plot")
        try:
            # Load the CSV file containing experiment results
            results_df = pd.read_csv("raw_results/results_binary.csv")
            # Expecting columns: 
            #   'n_estimators', 'max_depth', 'no_of_features',
            #   'train_accuracy', 'test_accuracy', 'selected_features'
            
            # Filter by test_accuracy and train_accuracy range
            filtered_df = results_df[
                (results_df['test_accuracy'] >= test_acc_range[0]) &
                (results_df['test_accuracy'] <= test_acc_range[1]) &
                (results_df['train_accuracy'] >= train_acc_range[0]) &
                (results_df['train_accuracy'] <= train_acc_range[1])
            ]
            
            # Further filter if any feature has been selected.
            if chosen_features:
                # Convert 'selected_features' column (string) to a list and check for matches
                def feature_match(row_features):
                    try:
                        feats = literal_eval(row_features)  # convert string to list
                        # Check if at least one chosen feature is in the selected features
                        return any(feat in feats for feat in chosen_features)
                    except:
                        return False
                filtered_df = filtered_df[filtered_df['selected_features'].apply(feature_match)]
            
            st.dataframe(filtered_df)
            
            # Plot: Accuracy vs n_estimators
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(filtered_df['n_estimators'], filtered_df['test_accuracy'], 
                       color='blue', label='Test Accuracy')
            ax.scatter(filtered_df['n_estimators'], filtered_df['train_accuracy'], 
                       color='green', label='Train Accuracy')
            ax.set_xlabel("n_estimators")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy vs n_estimators")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error("Error loading or filtering results_binary.csv file: " + str(e))

# -------------------------------------------------
# CUSTOM MODEL PAGE
# -------------------------------------------------
def custom_model_page():
    st.title("Custom Model Configuration & Results 🚀")
    st.subheader("Adjust the parameters to run your own model!")
    
    # Create two columns for parameters (left) and output (right)
    param_col, output_col = st.columns(2)
    
    with param_col:
        st.header("Model Parameters")
        custom_n_estimators = st.number_input("n_estimators", min_value=10, max_value=500, value=100, step=1)
        custom_max_depth = st.number_input("max_depth (0 for None)", min_value=0, max_value=100, value=0, step=1)
        custom_no_of_features = st.number_input("Number of Features", min_value=1, max_value=13, value=5, step=1)
        
        # Let the user choose a random state
        custom_random_state = st.number_input("Random State", min_value=0, max_value=1000, value=42, step=1)
        
        # Test size slider
        custom_test_size = st.slider("Test Size", 0.1, 0.5, 0.3, 0.01)
        
        # Optionally let user choose some features
        available_features = [
            'age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal'
        ]
        chosen_features = st.multiselect("Select specific features (optional):", available_features)
        
        run_custom = st.button("Run Custom Model")
    
    if run_custom:
        with output_col:
            st.header("Custom Model Results")
            try:
                # Ensure reproducibility with both Python's random and NumPy
                random.seed(custom_random_state)
                np.random.seed(custom_random_state)
                
                # Load the dataset (ensure file is available)
                data = pd.read_csv('processed.cleveland.data', header=None)
                all_columns = [
                    'age','sex','cp','trestbps','chol','fbs','restecg',
                    'thalach','exang','oldpeak','slope','ca','thal','target'
                ]
                data.columns = all_columns
                data = data[~data.isin(['?']).any(axis=1)]
                data['target'] = data['target'].replace([1,2,3,4], 1)
                
                # Determine features to use
                features_pool = available_features.copy()
                if chosen_features:
                    # If user-selected features are less than required, fill from the remaining
                    features = chosen_features.copy()
                    if len(features) < custom_no_of_features:
                        remaining = [f for f in features_pool if f not in features]
                        random.shuffle(remaining)
                        features.extend(remaining[: custom_no_of_features - len(features)])
                    else:
                        features = chosen_features[:custom_no_of_features]
                else:
                    # Shuffle and select randomly
                    random.shuffle(features_pool)
                    features = features_pool[:custom_no_of_features]
                
                st.write(f"**Selected Features:** {features}")
                
                # Split data
                x = data[features]
                y = data['target']
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=custom_test_size,
                    random_state=custom_random_state, stratify=y
                )
                
                # Scaling
                scaler = StandardScaler()
                x_train_scaled = scaler.fit_transform(x_train)
                x_test_scaled = scaler.transform(x_test)
                
                # Create and fit classifier
                clf = RandomForestClassifier(
                    n_estimators=custom_n_estimators,
                    max_depth=(None if custom_max_depth == 0 else custom_max_depth),
                    random_state=custom_random_state
                )
                clf.fit(x_train_scaled, y_train)
                y_pred = clf.predict(x_test_scaled)
                
                # Compute scores and metrics
                train_acc = accuracy_score(y_train, clf.predict(x_train_scaled))
                test_acc = accuracy_score(y_test, y_pred)
                class_rep = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # Show Accuracy
                st.write(f"**Train Accuracy:** {train_acc:.2f}")
                st.write(f"**Test Accuracy:** {test_acc:.2f}")
                
                # Convert classification report to a dataframe
                report_df = pd.DataFrame(class_rep).T
                # Keep only relevant columns if you like
                # report_df = report_df[['precision','recall','f1-score','support']]
                
                # Sensitivity (Recall) for positive class = class '1'
                sensitivity = report_df.loc['1', 'recall'] if '1' in report_df.index else None
                # Specificity = recall for class '0'
                specificity = report_df.loc['0', 'recall'] if '0' in report_df.index else None
                
                # Display classification report as a table
                st.markdown("### Classification Report (Table)")
                st.dataframe(report_df)
                
                # Display additional metrics
                if sensitivity is not None:
                    st.write(f"**Sensitivity (Recall for class 1):** {sensitivity:.2f}")
                if specificity is not None:
                    st.write(f"**Specificity (Recall for class 0):** {specificity:.2f}")
                
                # Plot confusion matrix with better labels
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(
                    conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=['Predicted 0','Predicted 1'],
                    yticklabels=['True 0','True 1']
                )
                ax_cm.set_xlabel("Predicted Label")
                ax_cm.set_ylabel("True Label")
                ax_cm.set_title("Confusion Matrix")
                st.pyplot(fig_cm)
                
            except Exception as e:
                st.error("Error running custom model: " + str(e))

# -------------------------------------------------
# MAIN ROUTING
# -------------------------------------------------
if page == "Home":
    home_page()
elif page == "Explore":
    explore_page()
elif page == "Custom Model":
    custom_model_page()
