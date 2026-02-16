import streamlit as st
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("churn_model.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
explainer = shap.TreeExplainer(model)

# ---------------- TITLE ----------------
st.title("Customer Churn Prediction System")
st.markdown("End-to-End ML System with Explainable and Business Interpretation")

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

with col2:
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    tech_support = st.selectbox("Tech Support?", ["No", "Yes"])
    paperless = st.selectbox("Paperless Billing?", ["No", "Yes"])

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Credit card (automatic)",
            "Bank transfer (automatic)"
        ]
    )

# ---------------- CREATE INPUT DATAFRAME ----------------
input_df = pd.DataFrame(columns=feature_columns)
input_df.loc[0] = 0

input_df["Tenure"] = tenure
input_df["MonthlyCharges"] = monthly_charges
input_df["TotalCharges"] = total_charges

if partner == "Yes":
    input_df["Partner_Yes"] = 1

if dependents == "Yes":
    input_df["Dependents_Yes"] = 1

if tech_support == "Yes":
    input_df["Tech Support_Yes"] = 1

if paperless == "Yes":
    input_df["Paperless Billing_Yes"] = 1

if contract == "One year":
    input_df["Contract_One year"] = 1
elif contract == "Two year":
    input_df["Contract_Two year"] = 1

if internet_service == "Fiber optic":
    input_df["Internet Service_Fiber optic"] = 1
elif internet_service == "No":
    input_df["Internet Service_No"] = 1

if payment_method == "Electronic check":
    input_df["Payment Method_Electronic check"] = 1
elif payment_method == "Mailed check":
    input_df["Payment Method_Mailed check"] = 1
elif payment_method == "Credit card (automatic)":
    input_df["Payment Method_Credit card (automatic)"] = 1

# ---------------- PREDICTION ----------------
if st.button("Predict Churn"):

    probability = model.predict_proba(input_df)[0][1]

    st.progress(float(probability))

    if probability >= 0.65:
        risk_level = "High"
        st.error(" High Risk of Churn")
    elif probability >= 0.30:
        risk_level = "Moderate"
        st.warning(" Moderate Risk of Churn")
    else:
        risk_level = "Low"
        st.success(" Low Risk of Churn")

    st.write(f"### Churn Probability: {probability:.2%}")

    # ---------------- SHAP EXPLANATION ----------------
    st.markdown("##  SHAP Explanation")

    shap_values = explainer(input_df)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    st.pyplot(fig)

    # ---------------- TEXT EXPLANATION ----------------
    st.markdown("---")
    st.markdown("## **Detailed Business Risk Report**")

    st.markdown(f"""
### **1️⃣ Executive Summary**

The customer has been classified as **{risk_level} Risk** with a predicted churn probability of **{probability:.2%}**.

This prediction is generated using a Random Forest machine learning model trained on historical telecom customer behavior patterns. The SHAP analysis above demonstrates how each feature contributed positively or negatively toward churn probability.
""")

    st.markdown("""
### **2️⃣ Tenure & Loyalty Evaluation**

Tenure reflects customer loyalty and switching barriers. Long-tenure customers typically show lower churn due to stronger brand attachment and higher transition costs. Low-tenure customers historically demonstrate higher churn probability because they lack long-term commitment.
""")

    st.markdown("""
### **3️⃣ Contract Commitment Assessment**

Long-term contracts significantly reduce churn risk because customers are bound by contractual obligations. Month-to-month contracts expose the company to higher churn vulnerability due to absence of commitment barriers.
""")

    st.markdown("""
### **4️⃣ Service & Technical Engagement Analysis**

Service type and technical support availability influence satisfaction. Premium services combined with dissatisfaction may increase churn. Access to tech support typically stabilizes customer retention.
""")

    st.markdown(f"""
### **5️⃣ Financial Contribution & Revenue Impact**

This customer contributes approximately **${monthly_charges:.2f} per month**, translating to an estimated annual revenue of **${monthly_charges * 12:.2f}**.

If this customer churns, the potential revenue loss equals this annual value. High-risk customers therefore represent significant financial exposure.
""")

    st.markdown("""
### **6️⃣ Behavioral & Household Stability Factors**

Customers with partners or dependents tend to exhibit longer retention patterns due to household stability. Independent customers may show higher mobility and switching tendencies.
""")

    st.markdown("""
### **7️⃣ Strategic Retention Recommendations**
""")

    if risk_level == "High":
        st.markdown("""
- Immediate personalized retention offer  
- Contract upgrade incentive  
- Proactive support engagement  
- Loyalty discount program  
""")
    elif risk_level == "Moderate":
        st.markdown("""
- Targeted promotional offers  
- Annual contract conversion strategy  
- Customer engagement campaigns  
""")
    else:
        st.markdown("""
- Maintain service quality  
- Offer loyalty rewards  
- Encourage referral programs  
""")

    st.markdown("""
### **8️⃣ Model Transparency & Responsible AI**

The Random Forest model was optimized using cross-validation and evaluated using ROC-AUC performance metrics. SHAP (SHapley Additive exPlanations) ensures interpretability by quantifying individual feature impact on predictions.

This approach aligns with responsible AI practices by making model decisions transparent and explainable.
""")
