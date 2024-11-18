# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Fetal Health Classification: A Machine Learning App')

# Display an image of mobile phones
st.image('fetal_health_image.gif', use_column_width=True, caption="Utilize advanced machine learning application to predict health classification!")

model_selection = None
uploaded_file = None

# Default DataFrame (used if no file is uploaded)
default_df = pd.read_csv('fetal_health.csv')
default_df.head()


# # Sidebar layout for file upload and model selection
# with st.sidebar:
#     # Step 1: Upload the file
#     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="Upload your CSV file with fetal health details.")

#     # Initialize variables for file preview and model selection
#     model_selection = None
#     df = None

#     # Step 2: Show the 'Preview' button only if a file is uploaded
#     if uploaded_file is not None:
#         preview_button = st.button("Preview File")
        
#         # Step 3: Show the file preview when the "Preview File" button is clicked
#         if preview_button:
#             df = pd.read_csv(uploaded_file)
#             st.write("### File Preview:")
#             st.dataframe(df.head())  # Display the first few rows of the dataframe
            
#             # Step 4: After previewing, show the model selection dropdown
#             model_selection = st.selectbox("Select which model you'd like to utilize:", 
#                                           ['Decision Tree', 'Random Forest', 'ADA Boost', 'Soft Voting'])
            
#             # Step 5: 'Predict' button to run the selected model
#             submit_button = st.button("Predict")
            
#             if submit_button:
#                 if df is not None and model_selection:
#                     st.write(f"Running prediction using the {model_selection} model...")
#                     # Add prediction logic here based on the model selected
#                 else:
#                     st.warning("Please upload a CSV file and select a model.")

#     else:
#         st.warning("Please upload a CSV file to proceed.")

# Sidebar layout for file upload and model selection
with st.sidebar:
    # Step 1: Upload the file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="Upload your CSV file with fetal health details.")

    # Initialize variables for file preview and model selection
    model_selection = None
    df = None

    # Display expected table format
    st.write("### Expected File Format")
    example_data = {
        "baseline value": [123, 123, 123],
        "accelerations": [0.004, 0, 0],
        "fetal_movement": [0, 0, 0],
        "uterine_contractions": [0.005, 0.005, 0.005],
        "light_decelerations": [0.005, 0.004, 0.005],
        "severe_decelerations": [0, 0, 0],
        "prolongued_decelerations": [0, 0, 0],
        "abnormal_short_term_variability": [24, 47, 50],
        "mean_value_of_short_term_variability": [1.3, 1.1, 0.8],
        "percentage_of_time_with_abnormal_long_term_variability": [0, 31, 32],
        "mean_value_of_long_term_variability": [2.1, 7.4, 3.1],
        "histogram_width": [56, 130, 94],
        "histogram_min": [92, 59, 75],
        "histogram_max": [148, 189, 169],
        "histogram_number_of_peaks": [3, 14, 7],
        "histogram_number_of_zeroes": [0, 2, 0],
        "histogram_mode": [121, 129, 125],
        "histogram_mean": [123, 122, 122],
        "histogram_median": [125, 127, 126],
        "histogram_variance": [7, 15, 8],
        "histogram_tendency": [0, 0, 0]
    }
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df)

    # Initialize session state for managing user actions
    if "df" not in st.session_state:
        st.session_state.df = None
    if "model_selection" not in st.session_state:
        st.session_state.model_selection = None

    # Step 2: Show the 'Preview' button only if a file is uploaded
    if uploaded_file is not None:
        if st.button("Preview File"):
            st.session_state.df = pd.read_csv(uploaded_file)
        
        # Step 3: Show the file preview if data is loaded
        if st.session_state.df is not None:
            st.write("### File Preview:")
            st.dataframe(st.session_state.df.head())  # Display the first few rows of the dataframe
            
            # Step 4: Model selection dropdown
            st.session_state.model_selection = st.selectbox(
                "Select which model you'd like to utilize:", 
                ['Decision Tree', 'Random Forest', 'ADA Boost', 'Soft Voting'], 
                index=0 if st.session_state.model_selection is None else
                      ['Decision Tree', 'Random Forest', 'ADA Boost', 'Soft Voting'].index(st.session_state.model_selection)
            )
            
            # Step 5: 'Predict' button to run the selected model
            if st.button("Predict"):
                if st.session_state.model_selection:
                    st.write(f"Running prediction using the {st.session_state.model_selection} model...")
                    # Add prediction logic here based on the model selected
                else:
                    st.warning("Please select a model.")

    else:
        st.warning("Please upload a CSV file to proceed.")

# Model loading based on selection
if st.session_state.model_selection:
    model_selection = st.session_state.model_selection
    if model_selection == 'Decision Tree':
        with open('decision_tree.pickle', 'rb') as dt_pickle:
            clf = pickle.load(dt_pickle)
    elif model_selection == 'Random Forest':
        with open('random_forest.pickle', 'rb') as rf_pickle:
            clf = pickle.load(rf_pickle)
    elif model_selection == 'ADA Boost':
        with open('ada_boost.pickle', 'rb') as ada_pickle:
            clf = pickle.load(ada_pickle)
    elif model_selection == 'Soft Voting':
        with open('soft_voting.pickle', 'rb') as sv_pickle:
            clf = pickle.load(sv_pickle)
    else:
        st.write("Please Select a Model")

if model_selection == 'Decision Tree':

    # Showing additional items in tabs
    st.subheader("Prediction Performance")
    tab1, tab2, tab3, tab4 = st.tabs(["Decision Tree", "Feature Importance", "Confusion Matrix", "Classification Report"])

    # Tab 1: Visualizing Decision Tree
    with tab1:
        st.write("### Decision Tree Visualization")
        st.image('dt.svg')
        st.caption("Visualization of the Decision Tree used in prediction.")

    # Tab 2: Feature Importance Visualization
    with tab2:
        st.write("### Feature Importance")
        st.image('dt_ft_import.svg')
        st.caption("Relative importance of features in prediction.")

    # Tab 3: Confusion Matrix
    with tab3:
        st.write("### Confusion Matrix")
        st.image('dt_cf_matrix.svg')
        st.caption("Confusion Matrix of model predictions.")

    # Tab 4: Classification Report
    with tab4:
        st.write("### Classification Report")
        report_df = pd.read_csv('dt_class_report.csv', index_col=0).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))

if model_selection == 'Random Forest':
    
    # Showing additional items in tabs
    st.subheader("Prediction Performance")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

     # Tab 1: Feature Importance Visualization
    with tab1:
        st.write("### Feature Importance")
        st.image('rf_ft_import.svg')
        st.caption("Relative importance of features in prediction.")

    # Tab 2: Confusion Matrix
    with tab2:
        st.write("### Confusion Matrix")
        st.image('rf_cf_matrix.svg')
        st.caption("Confusion Matrix of model predictions.")

    # Tab 3: Classification Report
    with tab3:
        st.write("### Classification Report")
        report_df = pd.read_csv('rf_class_report.csv', index_col=0).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))

if model_selection == 'ADA Boost':
    
    # Showing additional items in tabs
    st.subheader("Prediction Performance")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

    # Tab 1: Feature Importance Visualization
    with tab1:
        st.write("### Feature Importance")
        st.image('ada_ft_import.svg')
        st.caption("Relative importance of features in prediction.")

    # Tab 2: Confusion Matrix
    with tab2:
        st.write("### Confusion Matrix")
        st.image('ada_cf_matrix.svg')
        st.caption("Confusion Matrix of model predictions.")

    # Tab 3: Classification Report
    with tab3:
        st.write("### Classification Report")
        report_df = pd.read_csv('ada_class_report.csv', index_col=0).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))

if model_selection == 'Soft Voting':
    
    # Showing additional items in tabs
    st.subheader("Prediction Performance")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

   # Tab 1: Feature Importance Visualization
    with tab1:
        st.write("### Feature Importance")
        st.image('voting_ft_import.svg')
        st.caption("Relative importance of features in prediction.")

    # Tab 2: Confusion Matrix
    with tab2:
        st.write("### Confusion Matrix")
        st.image('voting_cf_matrix.svg')
        st.caption("Confusion Matrix of model predictions.")

    # Tab 3: Classification Report
    with tab3:
        st.write("### Classification Report")
        report_df = pd.read_csv('voting_class_report.csv', index_col=0).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))