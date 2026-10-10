import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from config import DATA_PATH, MODEL_PATH, METRICS_PATH
from src.data_utils import load_data, clean_data, data_quality
from src.ml_pipeline import train_models
from src.database import save_prediction, history

st.set_page_config(page_title="Customer Churn Analytics", layout="wide")
st.title("Customer Churn Prediction & Business Analytics System")
st.caption("Educational project: analyze churn patterns and estimate customer churn risk.")

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

raw = load_data(DATA_PATH)
df = clean_data(raw)
page = st.sidebar.radio("Choose page", [
    "Business Dashboard", "Predict Churn", "Data Explorer",
    "Model Performance", "Prediction History"
])

if page == "Business Dashboard":
    total = len(df)
    churned = int((df["Churn"].astype(str).str.lower() == "yes").sum())
    rate = 100 * churned / total if total else 0
    a, b, c, d = st.columns(4)
    a.metric("Total customers", f"{total:,}")
    b.metric("Churned customers", f"{churned:,}")
    c.metric("Retained customers", f"{total-churned:,}")
    d.metric("Historical churn rate", f"{rate:.1f}%")
    left, right = st.columns(2)
    with left:
        st.subheader("Churn distribution")
        fig, ax = plt.subplots()
        df["Churn"].value_counts().reindex(["No", "Yes"]).fillna(0).plot(kind="bar", ax=ax)
        ax.set_xlabel("Churn")
        ax.set_ylabel("Customers")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Churn rate by contract")
        rates = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
        if "Yes" in rates:
            fig, ax = plt.subplots()
            rates["Yes"].sort_values(ascending=False).plot(kind="bar", ax=ax)
            ax.set_ylabel("Churn rate (%)")
            st.pyplot(fig)
            plt.close(fig)
    st.subheader("Data quality snapshot")
    st.json(data_quality(raw))
    st.info("Historical churn metrics describe the dataset and do not guarantee future outcomes.")

elif page == "Predict Churn":
    st.subheader("Predict churn for a customer")
    st.write("Enter customer details. Other service fields use typical default values for this demo.")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            label = st.text_input("Customer label (optional)", placeholder="e.g. C1001")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.number_input("Monthly charges", 0.0, 1000.0, 70.0, 5.0)
            total_charges = st.number_input("Total charges", 0.0, 100000.0, float(tenure*monthly), 10.0)
        with c2:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            gender = st.selectbox("Gender", ["Female", "Male"])
        with c3:
            senior = st.selectbox("Senior citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            phone = st.selectbox("Phone service", ["Yes", "No"])
        submitted = st.form_submit_button("Predict churn risk", type="primary")
    if submitted:
        with st.spinner("Training model and predicting..."):
            results, best, model, features = train_models(df)
        sample = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": "No phone service" if phone == "No" else "No",
            "InternetService": internet,
            "OnlineSecurity": "No internet service" if internet == "No" else "No",
            "OnlineBackup": "No internet service" if internet == "No" else "No",
            "DeviceProtection": "No internet service" if internet == "No" else "No",
            "TechSupport": "No internet service" if internet == "No" else "No",
            "StreamingTV": "No internet service" if internet == "No" else "No",
            "StreamingMovies": "No internet service" if internet == "No" else "No",
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total_charges
        }
        input_df = pd.DataFrame([sample])
        for col in features:
            if col not in input_df.columns:
                input_df[col] = None
        input_df = input_df[features]
        probability = float(model.predict_proba(input_df)[0][1])
        prediction = "High risk" if probability >= 0.5 else "Low risk"
        x, y = st.columns(2)
        x.metric("Estimated churn probability", f"{probability:.1%}")
        y.metric("Risk category", prediction)
        if probability >= 0.5:
            st.warning("Consider checking whether this customer needs additional support.")
        else:
            st.success("The model estimates a lower churn risk for this customer.")
        save_prediction(label, prediction, probability)
        st.caption(f"Model: {best}. Prediction is an estimate, not a guarantee.")

elif page == "Data Explorer":
    st.subheader("Dataset explorer")
    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    st.dataframe(df.head(500), use_container_width=True)
    st.subheader("Missing values by column")
    st.dataframe(raw.isna().sum().rename("missing values"), use_container_width=True)
    st.subheader("Summary statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

elif page == "Model Performance":
    st.subheader("Compare machine learning models")
    if st.button("Train / retrain models", type="primary"):
        with st.spinner("Training models..."):
            results, best, _, _ = train_models(df)
        st.success(f"Best model by F1-score: {best}")
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            report = json.load(f)
        st.write("Best model:", report["best_model"])
        table = [{"Model": name, **metrics} for name, metrics in report["models"].items()]
        st.dataframe(pd.DataFrame(table).set_index("Model"), use_container_width=True)
    else:
        st.info("Click the button to train models and calculate metrics.")

elif page == "Prediction History":
    st.subheader("Saved prediction history")
    try:
        records = history()
        if records:
            hist = pd.DataFrame(records)
            st.dataframe(hist, use_container_width=True)
            st.download_button("Download history CSV", hist.to_csv(index=False),
                               file_name="prediction_history.csv", mime="text/csv")
        else:
            st.info("No predictions saved yet. Use the Predict Churn page first.")
    except Exception as e:
        st.error(f"Could not load database history: {e}")
