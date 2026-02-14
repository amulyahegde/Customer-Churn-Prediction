import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# ---------------- LOAD SAVED OBJECTS ----------------
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.title("Customer Churn Prediction System")

# ---------------- CREATE TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Home", "Dashboard", "Prediction", "Model Info"]
)

# ================= HOME TAB =================
with tab1:
    st.header("Project Overview")

    st.markdown("""
    ### About This Application
    
    This system predicts whether a customer is likely to churn 
    using a trained Machine Learning model.

    ### Key Features
    - Predicts churn probability
    - Displays churn risk percentage
    - Interactive analytics dashboard
    - Real-time customer prediction
    """)

    st.info("Built using Machine Learning and Streamlit")

# ================= DASHBOARD TAB =================
with tab2:
    st.header("Churn Analytics Dashboard")

    df = pd.read_csv("data/Churn.csv")

    # ---------------- KPI SECTION ----------------
    total_customers = len(df)
    churned = df["Churn"].value_counts().get("Yes", 0)
    churn_rate = (churned / total_customers) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", total_customers)
    col2.metric("Churned Customers", churned)
    col3.metric("Churn Rate (%)", f"{churn_rate:.2f}")

    st.divider()

    # ---------------- BUSINESS INSIGHTS ----------------
    st.subheader("Customer Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Churn Distribution")
        st.bar_chart(df["Churn"].value_counts())

    with col2:
        st.markdown("#### Contract Type vs Churn")
        contract_churn = pd.crosstab(df["Contract"], df["Churn"])
        st.bar_chart(contract_churn)

    st.divider()

    # ---------------- MODEL PERFORMANCE ----------------
    st.subheader("Model Performance")

    from sklearn.metrics import roc_curve, auc

    df_model = df.copy()

    if "customerID" in df_model.columns:
        df_model = df_model.drop("customerID", axis=1)

    # Convert target to numeric
    y = df_model["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
    X = df_model.drop("Churn", axis=1)

    # Encode and match training columns
    X_encoded = pd.get_dummies(X)
    X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale
    X_scaled = scaler.transform(X_encoded)

    # Predict probabilities
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Compute ROC
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    roc_df = pd.DataFrame({
        "False Positive Rate": fpr,
        "True Positive Rate": tpr
    })

    col1, col2 = st.columns([2, 1])

    with col1:
        st.line_chart(roc_df.set_index("False Positive Rate"))

    with col2:
        st.metric("AUC Score", f"{roc_auc:.2f}")
        st.caption("Higher AUC indicates better model discrimination ability.")
# ================= PREDICTION TAB =================
with tab3:
    st.header("Predict Customer Churn")

    df = pd.read_csv("data/Churn.csv")

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    input_data = {}

    col1, col2 = st.columns(2)

    features = df.drop("Churn", axis=1).columns

    for i, col in enumerate(features):
        if i % 2 == 0:
            container = col1
        else:
            container = col2

        if df[col].dtype == "object":
            input_data[col] = container.selectbox(col, df[col].unique())
        else:
            input_data[col] = container.number_input(col)

    input_df = pd.DataFrame([input_data])

    st.divider()

    if st.button("Predict"):

        # Encode input
        input_encoded = pd.get_dummies(input_df)

        # Match training columns exactly
        input_encoded = input_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Scale input
        input_scaled = scaler.transform(input_encoded)

        # Predict probability
        probability = model.predict_proba(input_scaled)
        risk = probability[0][1] * 100

        st.subheader("Prediction Result")

        if risk < 30:
            st.success(f"Low Risk of Churn ({risk:.2f}%)")
            st.info("Recommended Action: No immediate action required. Maintain regular engagement.")

        elif risk < 60:
            st.warning(f"Medium Risk of Churn ({risk:.2f}%)")
            st.info("Recommended Action: Offer promotional incentives or personalized engagement.")

        else:
            st.error(f"High Risk of Churn ({risk:.2f}%)")
            st.info("Recommended Action: Immediate retention call or special discount strategy.")

# ================= MODEL INFO TAB =================
with tab4:
    st.header("Model Information")

    st.markdown("""
    ### Model Details

    - Algorithm Used: Random Forest Classifier  
    - Target Variable: Churn  
    - Type: Binary Classification  
    - Output: Churn Probability Score  

    The system analyzes customer attributes and predicts
    the likelihood of churn to support retention strategies.
    """)
