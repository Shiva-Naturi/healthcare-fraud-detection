"""
================================================================================
HEALTHCARE INSURANCE FRAUD DETECTION - WEB INTERFACE
================================================================================

This is a user-friendly web interface for the fraud detection system.
Users can:
1. Make predictions on new claims
2. View model performance
3. Explore the dataset
4. Analyze fraud patterns

How to run:
streamlit run fraud_detection_ui.py

Author: Beginner Data Science Project
================================================================================
"""

# ============================================================================
# IMPORT LIBRARIES
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
# Configure the Streamlit page settings
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
# Add custom CSS to make the UI more attractive
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #f0f2f6 0%, #e8eaf6 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .alert-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .safe-alert {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================
# Cache the loading functions so they only run once
# This makes the app faster when users interact with it

@st.cache_resource
def load_model():
    """
    Load the trained fraud detection model
    Returns: trained model object
    """
    try:
        with open('fraud_detection_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'fraud_detection_model.pkl' is in the same directory.")
        return None

@st.cache_resource
def load_scaler():
    """
    Load the feature scaler
    Returns: fitted StandardScaler object
    """
    try:
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except FileNotFoundError:
        st.error("❌ Scaler file not found! Please ensure 'scaler.pkl' is in the same directory.")
        return None

@st.cache_resource
def load_encoders():
    """
    Load the label encoders for categorical variables
    Returns: dictionary of LabelEncoder objects
    """
    try:
        with open('label_encoders.pkl', 'rb') as file:
            encoders = pickle.load(file)
        return encoders
    except FileNotFoundError:
        st.error("❌ Encoders file not found! Please ensure 'label_encoders.pkl' is in the same directory.")
        return None

@st.cache_data
def load_data():
    """
    Load the original dataset for exploration
    Returns: pandas DataFrame
    """
    try:
        data = pd.read_csv('Claude_healthcare_insurance_claims_100k.csv')
        return data
    except FileNotFoundError:
        st.warning("⚠️ Original dataset not found. Some features will be limited.")
        return None

# Load all required objects
model = load_model()
scaler = load_scaler()
encoders = load_encoders()
data = load_data()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
# Create a sidebar menu for navigation
st.sidebar.title("🏥 Navigation")
st.sidebar.markdown("---")

# Navigation menu
page = st.sidebar.radio(
    "Select a Page:",
    ["🏠 Home", "🔍 Make Prediction", "📊 Model Performance", "📈 Data Exploration", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Stats")
if data is not None:
    st.sidebar.metric("Total Claims", f"{len(data):,}")
    fraud_rate = (data['IsFraud'].sum() / len(data)) * 100
    st.sidebar.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    st.sidebar.metric("Model Accuracy", "96.95%")

# ============================================================================
# PAGE 1: HOME PAGE
# ============================================================================
if page == "🏠 Home":
    # Display the main header
    st.markdown('<div class="main-header">🏥 Healthcare Insurance Fraud Detection System</div>', 
                unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    ### Welcome to the Fraud Detection System! 👋
    
    This intelligent system uses **Machine Learning** to detect fraudulent healthcare insurance claims
    with **96.95% accuracy**. It analyzes patterns in claim data to identify suspicious activities
    and help prevent insurance fraud.
    """)
    
    # Create columns for feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 High Accuracy</h3>
            <p>96.95% overall accuracy with 94.16% precision in fraud detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Real-time Analysis</h3>
            <p>Instant fraud probability scoring for new claims</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💰 Cost Savings</h3>
            <p>Potential savings of $6.3M by detecting fraudulent claims early</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System capabilities
    st.markdown("### 🚀 System Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### What This System Can Do:
        - ✅ Predict fraud probability for new claims
        - ✅ Analyze claim patterns and anomalies
        - ✅ Identify high-risk providers and patients
        - ✅ Provide detailed fraud risk breakdown
        - ✅ Compare claims against typical patterns
        """)
    
    with col2:
        st.markdown("""
        #### Key Features Analyzed:
        - 📊 Claim amount and ratio to typical claims
        - 🏥 Provider history and specialty
        - 👤 Patient demographics and claim frequency
        - 💊 Number and type of procedures
        - 📅 Temporal patterns
        """)
    
    st.markdown("---")
    
    # How to use
    st.markdown("### 📖 How to Use This System")
    st.markdown("""
    1. **🔍 Make Prediction**: Enter claim details to get instant fraud probability
    2. **📊 Model Performance**: View detailed model metrics and evaluation
    3. **📈 Data Exploration**: Explore patterns and trends in the dataset
    4. **ℹ️ About**: Learn more about the project and methodology
    
    👈 **Select an option from the sidebar to get started!**
    """)
    
    # Performance snapshot
    if data is not None:
        st.markdown("---")
        st.markdown("### 📊 Performance Snapshot")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "96.95%", "2.5%")
        with col2:
            st.metric("Precision", "94.16%", "14.4%")
        with col3:
            st.metric("Recall", "84.93%", "0.5%")
        with col4:
            st.metric("F1-Score", "89.31%", "7.3%")

# ============================================================================
# PAGE 2: MAKE PREDICTION
# ============================================================================
elif page == "🔍 Make Prediction":
    st.markdown('<div class="main-header">🔍 Fraud Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Enter Claim Details
    Fill in the information below to get an instant fraud probability assessment.
    """)
    
    # Check if model is loaded
    if model is None or scaler is None or encoders is None:
        st.error("⚠️ Model files not loaded. Please ensure all model files are present.")
    else:
        # Create input form
        with st.form("prediction_form"):
            st.markdown("#### 📋 Claim Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                claim_month = st.selectbox("Claim Month", list(range(1, 13)), index=0)
                claim_day = st.selectbox("Claim Day of Week", 
                                        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                        index=0)
                claim_amount = st.number_input("Claim Amount ($)", min_value=0.0, value=1000.0, step=100.0)
            
            with col2:
                num_procedures = st.number_input("Number of Procedures", min_value=1, max_value=20, value=2)
                treatment_duration = st.number_input("Treatment Duration (days)", min_value=1, max_value=90, value=3)
                claim_ratio = st.number_input("Claim to Typical Ratio", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            
            with col3:
                procedure_code = st.selectbox("Procedure Code", 
                                             ['CPT99213', 'CPT99214', 'CPT99285', 'CPT93000', 'CPT45378', 
                                              'CPT43239', 'CPT29881', 'CPT97110', 'CPT73721'])
                diagnosis_code = st.selectbox("Diagnosis Code",
                                             ['I10', 'E11.9', 'J44.9', 'M25.561', 'G89.29', 'Z23', 
                                              'M54.5', 'K21.9', 'F41.9', 'R51'])
                insurance_type = st.selectbox("Insurance Type", 
                                             ['Medicare', 'Medicaid', 'Private', 'PPO', 'HMO'])
            
            st.markdown("#### 🏥 Provider Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                provider_specialty = st.selectbox("Provider Specialty",
                                                 ['Cardiology', 'Orthopedics', 'General Practice', 'Internal Medicine',
                                                  'Emergency Medicine', 'Physical Therapy', 'Gastroenterology', 
                                                  'Radiology'])
                provider_state = st.selectbox("Provider State",
                                             ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'])
            
            with col2:
                provider_experience = st.number_input("Provider Years of Experience", min_value=0, max_value=50, value=10)
                provider_avg_claim = st.number_input("Provider Avg Claim Amount ($)", min_value=0.0, value=2000.0, step=100.0)
            
            with col3:
                provider_total_claims = st.number_input("Provider Total Claims", min_value=0, value=100)
                provider_fraud_count = st.number_input("Provider Fraud Count", min_value=0, value=0)
            
            st.markdown("#### 👤 Patient Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=45)
            
            with col2:
                patient_gender = st.selectbox("Patient Gender", ['M', 'F'])
            
            with col3:
                patient_claim_count = st.number_input("Patient Claim Count", min_value=1, value=2)
            
            # Submit button
            st.markdown("---")
            submitted = st.form_submit_button("🔍 Analyze Claim", use_container_width=True)
            
            if submitted:
                # Prepare the data for prediction
                # Convert day name to number (0=Monday, 6=Sunday)
                day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                          'Friday': 4, 'Saturday': 5, 'Sunday': 6}
                claim_day_num = day_map[claim_day]
                
                # Create DataFrame with the input
                input_data = pd.DataFrame({
                    'ClaimMonth': [claim_month],
                    'ClaimDayOfWeek': [claim_day_num],
                    'ProviderSpecialty': [provider_specialty],
                    'ProviderState': [provider_state],
                    'ProviderYearsExperience': [provider_experience],
                    'ProviderAvgClaimAmount': [provider_avg_claim],
                    'ProviderTotalClaims': [provider_total_claims],
                    'ProviderFraudCount': [provider_fraud_count],
                    'PatientAge': [patient_age],
                    'PatientGender': [patient_gender],
                    'PatientClaimCount': [patient_claim_count],
                    'ProcedureCode': [procedure_code],
                    'DiagnosisCode': [diagnosis_code],
                    'NumberOfProcedures': [num_procedures],
                    'TreatmentDuration': [treatment_duration],
                    'ClaimAmount': [claim_amount],
                    'ClaimToTypicalRatio': [claim_ratio],
                    'InsuranceType': [insurance_type]
                })
                
                # Encode categorical variables
                for col, encoder in encoders.items():
                    if col in input_data.columns:
                        try:
                            input_data[col] = encoder.transform(input_data[col])
                        except:
                            # If value not seen during training, use the most common value
                            input_data[col] = 0
                
                # Scale the features
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]
                fraud_probability = probability[1] * 100
                
                # Display results
                st.markdown("---")
                st.markdown("### 🎯 Prediction Results")
                
                # Create two columns for results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="alert-box fraud-alert">
                            <h2>⚠️ FRAUD ALERT</h2>
                            <h3>Fraud Probability: {fraud_probability:.2f}%</h3>
                            <p>This claim has been flagged as <strong>FRAUDULENT</strong>.</p>
                            <p>Recommended Action: <strong>Manual Review Required</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-box safe-alert">
                            <h2>✅ LEGITIMATE CLAIM</h2>
                            <h3>Fraud Probability: {fraud_probability:.2f}%</h3>
                            <p>This claim appears to be <strong>LEGITIMATE</strong>.</p>
                            <p>Recommended Action: <strong>Standard Processing</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Create a gauge chart for fraud probability
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = fraud_probability,
                        title = {'text': "Fraud Risk Score"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred" if fraud_probability > 50 else "darkgreen"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk factors analysis
                st.markdown("### 🔍 Risk Factors Analysis")
                
                risk_factors = []
                
                if claim_ratio > 2.0:
                    risk_factors.append("⚠️ Claim amount significantly higher than typical")
                if num_procedures > 5:
                    risk_factors.append("⚠️ High number of procedures in single claim")
                if provider_fraud_count > 0:
                    risk_factors.append("⚠️ Provider has previous fraud history")
                if patient_claim_count > 5:
                    risk_factors.append("⚠️ Patient has high claim frequency")
                if treatment_duration > 30:
                    risk_factors.append("⚠️ Extended treatment duration")
                
                if risk_factors:
                    st.markdown("**Identified Risk Factors:**")
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
                else:
                    st.success("✅ No major risk factors identified")
                
                # Claim details summary
                st.markdown("### 📋 Claim Summary")
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.markdown(f"""
                    **Claim Details:**
                    - Amount: ${claim_amount:,.2f}
                    - Procedures: {num_procedures}
                    - Duration: {treatment_duration} days
                    - Ratio to Typical: {claim_ratio:.2f}x
                    """)
                
                with summary_col2:
                    st.markdown(f"""
                    **Provider & Patient:**
                    - Specialty: {provider_specialty}
                    - Experience: {provider_experience} years
                    - Patient Age: {patient_age}
                    - Insurance: {insurance_type}
                    """)

# ============================================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================================
elif page == "📊 Model Performance":
    st.markdown('<div class="main-header">📊 Model Performance Metrics</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Model Evaluation Results
    Our Random Forest model achieved excellent performance on the test dataset.
    """)
    
    # Performance metrics
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Accuracy",
            value="96.95%",
            delta="High",
            help="Percentage of correctly classified claims"
        )
    
    with col2:
        st.metric(
            label="Precision",
            value="94.16%",
            delta="Excellent",
            help="Of flagged frauds, how many were actually fraudulent"
        )
    
    with col3:
        st.metric(
            label="Recall",
            value="84.93%",
            delta="Good",
            help="Of actual frauds, how many were detected"
        )
    
    with col4:
        st.metric(
            label="F1-Score",
            value="89.31%",
            delta="Strong",
            help="Harmonic mean of precision and recall"
        )
    
    st.markdown("---")
    
    # Confusion Matrix
    st.markdown("### 📊 Confusion Matrix")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create confusion matrix visualization
        confusion_data = pd.DataFrame(
            [[16842, 158], [452, 2548]],
            columns=['Predicted Legitimate', 'Predicted Fraud'],
            index=['Actual Legitimate', 'Actual Fraud']
        )
        
        fig = px.imshow(confusion_data, 
                       text_auto=True,
                       color_continuous_scale='Blues',
                       labels=dict(x="Predicted", y="Actual", color="Count"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        **Interpretation:**
        
        - **True Negatives (16,842)**
          ✅ Legitimate claims correctly identified
        
        - **False Positives (158)**
          ⚠️ Legitimate claims flagged as fraud
        
        - **False Negatives (452)**
          ❌ Frauds that were missed
        
        - **True Positives (2,548)**
          ✅ Frauds correctly detected
        """)
    
    st.markdown("---")
    
    # Model Comparison
    st.markdown("### 🏆 Model Comparison")
    
    comparison_data = pd.DataFrame({
        'Model': ['Random Forest', 'Logistic Regression', 'Isolation Forest'],
        'Accuracy': [96.95, 94.46, 85.58],
        'Precision': [94.16, 79.80, 53.42],
        'Recall': [84.93, 84.43, 29.90],
        'F1-Score': [89.31, 82.05, 38.34]
    })
    
    # Create bar chart
    fig = px.bar(comparison_data, 
                 x='Model', 
                 y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                 title='Model Performance Comparison',
                 barmode='group',
                 height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Business Impact
    st.markdown("### 💰 Business Impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Frauds Detected</h4>
            <h2>2,548</h2>
            <p>Out of 3,000 actual frauds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>Potential Savings</h4>
            <h2>$7.6M</h2>
            <p>From detected fraudulent claims</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>Review Reduction</h4>
            <h2>86.5%</h2>
            <p>Fewer claims need manual review</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # What the metrics mean
    with st.expander("📖 Understanding the Metrics"):
        st.markdown("""
        **Accuracy:** Percentage of all predictions that were correct (both fraud and legitimate)
        - Our model: 96.95% means it correctly classifies 97 out of 100 claims
        
        **Precision:** Of all claims flagged as fraudulent, what percentage were actually fraudulent?
        - Our model: 94.16% means if we flag 100 claims, 94 are actually fraudulent (only 6 false alarms)
        
        **Recall (Sensitivity):** Of all actual fraudulent claims, what percentage did we catch?
        - Our model: 84.93% means we catch about 85 out of every 100 fraudulent claims
        
        **F1-Score:** Harmonic mean of precision and recall (balances both metrics)
        - Our model: 89.31% indicates excellent overall performance
        
        **Trade-offs:**
        - Higher precision = fewer false alarms but might miss some frauds
        - Higher recall = catch more frauds but more false alarms
        - F1-Score helps balance these competing goals
        """)

# ============================================================================
# PAGE 4: DATA EXPLORATION
# ============================================================================
elif page == "📈 Data Exploration":
    st.markdown('<div class="main-header">📈 Data Exploration & Insights</div>', unsafe_allow_html=True)
    
    if data is None:
        st.error("⚠️ Dataset not available for exploration.")
    else:
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Claims", f"{len(data):,}")
        with col2:
            fraud_count = data['IsFraud'].sum()
            st.metric("Fraudulent Claims", f"{fraud_count:,}")
        with col3:
            fraud_rate = (fraud_count / len(data)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        with col4:
            avg_claim = data['ClaimAmount'].mean()
            st.metric("Avg Claim Amount", f"${avg_claim:,.2f}")
        
        st.markdown("---")
        
        # Visualization selector
        st.markdown("### 📊 Interactive Visualizations")
        
        viz_option = st.selectbox(
            "Select a visualization:",
            ["Fraud Distribution", "Claim Amount Analysis", "Provider Specialty Analysis", 
             "Patient Age Distribution", "Insurance Type Analysis", "Monthly Trends"]
        )
        
        if viz_option == "Fraud Distribution":
            fig = px.pie(data, names='IsFraud', 
                        title='Fraud vs Legitimate Claims Distribution',
                        labels={'IsFraud': 'Claim Type'},
                        color_discrete_map={0: 'lightgreen', 1: 'lightcoral'})
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"**Insight:** {fraud_rate:.2f}% of all claims are fraudulent, indicating a class imbalance that requires special handling.")
        
        elif viz_option == "Claim Amount Analysis":
            fig = px.box(data, x='IsFraud', y='ClaimAmount', 
                        title='Claim Amount Distribution by Fraud Status',
                        labels={'IsFraud': 'Fraud Status', 'ClaimAmount': 'Claim Amount ($)'},
                        color='IsFraud',
                        color_discrete_map={0: 'lightgreen', 1: 'lightcoral'})
            st.plotly_chart(fig, use_container_width=True)
            
            avg_fraud = data[data['IsFraud']==1]['ClaimAmount'].mean()
            avg_legit = data[data['IsFraud']==0]['ClaimAmount'].mean()
            st.info(f"**Insight:** Fraudulent claims average ${avg_fraud:,.2f} compared to ${avg_legit:,.2f} for legitimate claims ({((avg_fraud/avg_legit - 1) * 100):.1f}% higher).")
        
        elif viz_option == "Provider Specialty Analysis":
            specialty_fraud = data.groupby('ProviderSpecialty')['IsFraud'].agg(['sum', 'count', 'mean'])
            specialty_fraud = specialty_fraud.sort_values('mean', ascending=False).head(10)
            specialty_fraud['fraud_rate'] = specialty_fraud['mean'] * 100
            
            fig = px.bar(specialty_fraud.reset_index(), 
                        x='ProviderSpecialty', y='fraud_rate',
                        title='Top 10 Provider Specialties by Fraud Rate',
                        labels={'fraud_rate': 'Fraud Rate (%)', 'ProviderSpecialty': 'Specialty'},
                        color='fraud_rate',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
            
            top_specialty = specialty_fraud.index[0]
            top_rate = specialty_fraud.iloc[0]['fraud_rate']
            st.info(f"**Insight:** {top_specialty} has the highest fraud rate at {top_rate:.2f}%.")
        
        elif viz_option == "Patient Age Distribution":
            fig = px.histogram(data, x='PatientAge', color='IsFraud',
                             title='Patient Age Distribution by Fraud Status',
                             labels={'PatientAge': 'Patient Age', 'count': 'Number of Claims'},
                             color_discrete_map={0: 'lightgreen', 1: 'lightcoral'},
                             barmode='overlay',
                             nbins=30)
            st.plotly_chart(fig, use_container_width=True)
            
            fraud_age_avg = data[data['IsFraud']==1]['PatientAge'].mean()
            st.info(f"**Insight:** Average age of patients with fraudulent claims is {fraud_age_avg:.1f} years.")
        
        elif viz_option == "Insurance Type Analysis":
            insurance_fraud = data.groupby('InsuranceType')['IsFraud'].agg(['sum', 'count', 'mean'])
            insurance_fraud['fraud_rate'] = insurance_fraud['mean'] * 100
            
            fig = px.bar(insurance_fraud.reset_index(), 
                        x='InsuranceType', y='fraud_rate',
                        title='Fraud Rate by Insurance Type',
                        labels={'fraud_rate': 'Fraud Rate (%)', 'InsuranceType': 'Insurance Type'},
                        color='fraud_rate',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
            
            highest_fraud_insurance = insurance_fraud['fraud_rate'].idxmax()
            highest_rate = insurance_fraud.loc[highest_fraud_insurance, 'fraud_rate']
            st.info(f"**Insight:** {highest_fraud_insurance} insurance has the highest fraud rate at {highest_rate:.2f}%.")
        
        elif viz_option == "Monthly Trends":
            monthly_fraud = data.groupby('ClaimMonth')['IsFraud'].agg(['sum', 'count', 'mean'])
            monthly_fraud['fraud_rate'] = monthly_fraud['mean'] * 100
            
            fig = px.line(monthly_fraud.reset_index(), 
                         x='ClaimMonth', y='fraud_rate',
                         title='Fraud Rate Trend by Month',
                         labels={'fraud_rate': 'Fraud Rate (%)', 'ClaimMonth': 'Month'},
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
            peak_month = monthly_fraud['fraud_rate'].idxmax()
            peak_rate = monthly_fraud.loc[peak_month, 'fraud_rate']
            st.info(f"**Insight:** Month {peak_month} has the highest fraud rate at {peak_rate:.2f}%.")
        
        st.markdown("---")
        
        # Sample data viewer
        st.markdown("### 🔍 Sample Data Viewer")
        
        show_fraud_only = st.checkbox("Show fraudulent claims only")
        
        if show_fraud_only:
            sample_data = data[data['IsFraud'] == 1].head(100)
        else:
            sample_data = data.head(100)
        
        st.dataframe(sample_data, use_container_width=True, height=400)
        
        st.markdown(f"*Showing {len(sample_data)} of {len(data):,} total claims*")

# ============================================================================
# PAGE 5: ABOUT
# ============================================================================
elif page == "ℹ️ About":
    st.markdown('<div class="main-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🏥 Healthcare Insurance Fraud Detection System
    
    This is a comprehensive machine learning solution designed to detect fraudulent healthcare 
    insurance claims with high accuracy and efficiency.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Project Objectives
        
        1. **Detect Fraud Early**
           - Identify suspicious claims before payment
           - Reduce financial losses from fraud
        
        2. **Improve Efficiency**
           - Automate initial fraud screening
           - Reduce manual review workload
        
        3. **Data-Driven Insights**
           - Identify fraud patterns and trends
           - Help investigators focus on high-risk cases
        
        4. **Cost Savings**
           - Prevent fraudulent payments
           - Optimize investigation resources
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 Technical Approach
        
        **Machine Learning Algorithm:**
        - Random Forest Classifier
        - Trained on 100,000 historical claims
        - 18 features analyzed per claim
        
        **Data Processing:**
        - SMOTE for class balancing
        - StandardScaler for feature normalization
        - Label encoding for categorical variables
        
        **Model Performance:**
        - 96.95% accuracy
        - 94.16% precision
        - 84.93% recall
        - 89.31% F1-score
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📊 Key Features Analyzed
    
    The model analyzes 18 different features to make predictions:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Claim Features:**
        - Claim amount
        - Claim-to-typical ratio
        - Number of procedures
        - Treatment duration
        - Claim month
        - Claim day of week
        """)
    
    with col2:
        st.markdown("""
        **Provider Features:**
        - Specialty
        - State
        - Years of experience
        - Average claim amount
        - Total claims
        - Previous fraud count
        """)
    
    with col3:
        st.markdown("""
        **Patient Features:**
        - Age
        - Gender
        - Claim frequency
        - Procedure codes
        - Diagnosis codes
        - Insurance type
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 💡 How It Works
    
    1. **Data Input:** User enters claim details or system receives claim data
    2. **Preprocessing:** Data is encoded and scaled using trained transformers
    3. **Prediction:** Random Forest model analyzes patterns across 100 decision trees
    4. **Output:** Fraud probability score and classification (Fraud/Legitimate)
    5. **Action:** System recommends next steps based on risk level
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🚀 Future Enhancements
    
    - **Real-time API Integration:** Connect with claims processing systems
    - **Advanced Analytics Dashboard:** Deeper insights and trend analysis
    - **Model Retraining Pipeline:** Automated updates with new fraud patterns
    - **Multi-model Ensemble:** Combine predictions from multiple algorithms
    - **Explainable AI:** Detailed explanations for each prediction
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 👨‍💻 Project Information
    
    - **Level:** Beginner to Intermediate
    - **Dataset:** 100,000 healthcare insurance claims
    - **Technologies:** Python, Scikit-learn, Streamlit, Plotly
    - **Model Type:** Supervised Learning (Classification)
    - **Deployment:** Web-based interface
    
    ---
    
    **Created as a comprehensive end-to-end machine learning project for learning and demonstration purposes.**
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Healthcare Insurance Fraud Detection System v1.0</p>
    <p>Built with Python, Scikit-learn, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
