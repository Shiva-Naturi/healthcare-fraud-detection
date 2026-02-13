# 🌐 Web UI User Guide - Fraud Detection System

## 🚀 How to Launch the Web Interface

### Step 1: Install Streamlit
```bash
pip install streamlit plotly --break-system-packages
```

### Step 2: Navigate to Project Directory
```bash
cd /path/to/your/project/folder
```

Make sure this folder contains:
- `fraud_detection_ui.py` (the UI file)
- `fraud_detection_model.pkl` (trained model)
- `scaler.pkl` (feature scaler)
- `label_encoders.pkl` (encoders)
- Access to the dataset CSV file

### Step 3: Launch the Application
```bash
streamlit run fraud_detection_ui.py
```

### Step 4: Access the Interface
The browser will automatically open to: `http://localhost:8501`

If it doesn't open automatically, manually visit that URL in your browser.

---

## 📱 User Interface Overview

### 🏠 Home Page
**What you'll see:**
- Welcome message and system overview
- Key performance metrics (Accuracy, Precision, Recall, F1-Score)
- System capabilities and features
- Quick performance snapshot

**What you can do:**
- Get an overview of the system
- See performance highlights
- Understand what the system can do

---

### 🔍 Make Prediction Page
**What you'll see:**
- Interactive form to enter claim details
- Three sections: Claim Info, Provider Info, Patient Info

**What you can do:**

#### Example: Predicting a Fraud Case

1. **Fill in Claim Information:**
   - Claim Month: `6` (June)
   - Claim Day of Week: `Tuesday`
   - Claim Amount: `$8,500`
   - Number of Procedures: `8`
   - Treatment Duration: `45 days`
   - Claim to Typical Ratio: `3.5`
   - Procedure Code: `CPT43239`
   - Diagnosis Code: `E11.9`
   - Insurance Type: `Private`

2. **Fill in Provider Information:**
   - Provider Specialty: `Gastroenterology`
   - Provider State: `CA`
   - Provider Years of Experience: `5`
   - Provider Avg Claim Amount: `$3,200`
   - Provider Total Claims: `500`
   - Provider Fraud Count: `3`

3. **Fill in Patient Information:**
   - Patient Age: `52`
   - Patient Gender: `M`
   - Patient Claim Count: `8`

4. **Click "Analyze Claim"**

**Result:**
- You'll see a FRAUD ALERT with probability score
- Gauge chart showing risk level
- List of identified risk factors
- Recommended action (Manual Review Required)

#### Example: Predicting a Legitimate Case

1. **Fill in Claim Information:**
   - Claim Month: `3`
   - Claim Day of Week: `Monday`
   - Claim Amount: `$1,200`
   - Number of Procedures: `2`
   - Treatment Duration: `3 days`
   - Claim to Typical Ratio: `1.1`
   - Procedure Code: `CPT99213`
   - Diagnosis Code: `I10`
   - Insurance Type: `Medicare`

2. **Fill in Provider Information:**
   - Provider Specialty: `General Practice`
   - Provider State: `NY`
   - Provider Years of Experience: `15`
   - Provider Avg Claim Amount: `$1,800`
   - Provider Total Claims: `1,200`
   - Provider Fraud Count: `0`

3. **Fill in Patient Information:**
   - Patient Age: `68`
   - Patient Gender: `F`
   - Patient Claim Count: `2`

4. **Click "Analyze Claim"**

**Result:**
- You'll see LEGITIMATE CLAIM confirmation
- Low fraud probability score (< 30%)
- No major risk factors identified
- Recommended action (Standard Processing)

---

### 📊 Model Performance Page
**What you'll see:**
- Key performance metrics (Accuracy, Precision, Recall, F1-Score)
- Interactive confusion matrix
- Model comparison charts
- Business impact metrics

**What you can do:**
- Understand how well the model performs
- See the confusion matrix breakdown
- Compare different models (Random Forest, Logistic Regression, Isolation Forest)
- View financial impact estimates
- Expand "Understanding the Metrics" section for explanations

**Key Insights:**
- Random Forest is the best performing model
- 96.95% accuracy means 97 out of 100 predictions are correct
- 94.16% precision means only 6 false alarms per 100 flagged claims
- Potential savings of $7.6M from detected frauds

---

### 📈 Data Exploration Page
**What you'll see:**
- Dataset overview metrics
- Interactive visualization selector
- Sample data viewer

**What you can do:**

1. **Select Different Visualizations:**
   - **Fraud Distribution:** Pie chart showing fraud vs legitimate ratio
   - **Claim Amount Analysis:** Box plot comparing claim amounts
   - **Provider Specialty Analysis:** Bar chart of fraud rates by specialty
   - **Patient Age Distribution:** Histogram of patient ages
   - **Insurance Type Analysis:** Fraud rates by insurance type
   - **Monthly Trends:** Line chart showing fraud trends over months

2. **View Sample Data:**
   - Toggle "Show fraudulent claims only" to filter
   - Browse first 100 claims in a table
   - Scroll through different columns

**Example Insights:**
- Gastroenterology has highest fraud rate (~15%)
- Fraudulent claims average 40% higher than legitimate
- Fraud patterns vary by month
- Certain insurance types show higher fraud rates

---

### ℹ️ About Page
**What you'll see:**
- Project objectives and goals
- Technical approach details
- Features analyzed by the model
- How the system works
- Future enhancement plans

**What you can do:**
- Learn about the project background
- Understand the technical methodology
- See what features are important for prediction
- Explore future development plans

---

## 💡 Use Cases & Examples

### Use Case 1: Daily Claim Screening
**Scenario:** Insurance company receives 500 new claims daily

**Workflow:**
1. Go to "Make Prediction" page
2. For each claim, enter the details
3. System provides instant fraud probability
4. Claims with >50% probability → Manual review queue
5. Claims with <50% probability → Standard processing

**Result:** 86.5% reduction in manual review workload

---

### Use Case 2: Investigating a Suspicious Provider
**Scenario:** Provider has submitted unusually high claims

**Workflow:**
1. Go to "Data Exploration" page
2. Select "Provider Specialty Analysis"
3. Check if the specialty has high fraud rate
4. Go to "Make Prediction"
5. Enter provider details with ProviderFraudCount > 0
6. System flags high risk

**Result:** Targeted investigation of high-risk providers

---

### Use Case 3: Monthly Fraud Pattern Analysis
**Scenario:** Management wants to understand fraud trends

**Workflow:**
1. Go to "Data Exploration" page
2. Select "Monthly Trends" visualization
3. Identify months with peak fraud rates
4. Plan additional scrutiny during high-risk months

**Result:** Proactive fraud prevention during peak periods

---

### Use Case 4: Training New Fraud Investigators
**Scenario:** New staff need to learn fraud indicators

**Workflow:**
1. Start at "Home" page for overview
2. Go to "About" page to understand methodology
3. Use "Make Prediction" with example cases
4. View "Model Performance" to understand accuracy
5. Explore "Data Exploration" to see patterns

**Result:** Well-trained staff who understand fraud indicators

---

## 🎯 Tips for Best Results

### For Accurate Predictions:
1. **Enter Complete Information:** Fill all fields accurately
2. **Use Realistic Values:** Match the ranges in your actual data
3. **Check Provider History:** ProviderFraudCount is very important
4. **Consider Claim Ratio:** Values > 2.0 are high risk
5. **Multiple Procedures:** Claims with >5 procedures are suspicious

### Understanding Risk Levels:
- **0-30% probability:** Low risk - Standard processing
- **30-50% probability:** Medium risk - Additional verification
- **50-70% probability:** High risk - Manual review recommended
- **70-100% probability:** Critical risk - Thorough investigation required

### Common Red Flags:
- Claim amount >> typical for that provider
- Provider has previous fraud history (FraudCount > 0)
- Patient has many claims (>5)
- Excessive number of procedures (>8)
- Very high ClaimToTypicalRatio (>2.5)

---

## 🔧 Troubleshooting

### Problem: UI doesn't load
**Solution:**
```bash
# Make sure Streamlit is installed
pip install streamlit plotly --break-system-packages

# Check if port 8501 is available
# If not, use a different port:
streamlit run fraud_detection_ui.py --server.port 8502
```

### Problem: "Model file not found" error
**Solution:**
- Ensure `fraud_detection_model.pkl` is in the same folder as `fraud_detection_ui.py`
- Check file permissions
- Verify the model was saved correctly

### Problem: Prediction gives error
**Solution:**
- Ensure all required fields are filled
- Check that values are within reasonable ranges
- Verify the model files are not corrupted

### Problem: Data Exploration page shows "Dataset not available"
**Solution:**
- Update the CSV file path in line 134 of `fraud_detection_ui.py`
- Make sure the CSV file is accessible
- Check file permissions

---

## 📊 Understanding the Output

### Fraud Prediction Output Explained:

**FRAUD ALERT Example:**
```
⚠️ FRAUD ALERT
Fraud Probability: 87.35%
This claim has been flagged as FRAUDULENT.
Recommended Action: Manual Review Required

Risk Factors:
⚠️ Claim amount significantly higher than typical
⚠️ High number of procedures in single claim
⚠️ Provider has previous fraud history
```

**What this means:**
- The model is 87% confident this is fraud
- Multiple risk factors identified
- Human investigator should review before payment

**LEGITIMATE CLAIM Example:**
```
✅ LEGITIMATE CLAIM
Fraud Probability: 12.45%
This claim appears to be LEGITIMATE.
Recommended Action: Standard Processing

✅ No major risk factors identified
```

**What this means:**
- Only 12% chance of fraud (low risk)
- No significant red flags
- Can proceed with normal processing

---

## 🎓 Learning Exercises

### Exercise 1: Test Different Scenarios
Create claims with these characteristics and see predictions:
1. Normal claim (low values, no history)
2. High-value claim (5x typical)
3. Many procedures (>10)
4. Provider with fraud history
5. Combination of all red flags

### Exercise 2: Explore Patterns
1. Go to Data Exploration
2. Try all 6 visualizations
3. Write down 3 insights you discover

### Exercise 3: Compare Models
1. Go to Model Performance
2. Note the metrics for each model
3. Explain why Random Forest is best

---

## 🚀 Advanced Features

### Customizing the UI:

**Change Colors:**
Edit lines 30-60 in `fraud_detection_ui.py` to modify CSS styles

**Add New Visualizations:**
Add new options to the visualization selector (lines 500+)

**Modify Threshold:**
Change the 50% decision threshold to be more/less conservative

**Add New Metrics:**
Include additional business metrics on the Performance page

---

## 📞 Quick Reference

### Navigation:
- **Home:** Overview and introduction
- **Make Prediction:** Enter claim details, get fraud score
- **Model Performance:** View accuracy metrics and comparisons
- **Data Exploration:** Interactive charts and insights
- **About:** Technical details and methodology

### Key Shortcuts:
- `R` - Refresh/rerun the app
- `C` - Clear cache
- `Ctrl+C` - Stop the server

### Important Files:
- `fraud_detection_ui.py` - Main UI application
- `fraud_detection_model.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `label_encoders.pkl` - Category encoders

---

**Enjoy using the Fraud Detection System! 🎉**

For questions or issues, refer to the main README.md file or check the troubleshooting section above.
