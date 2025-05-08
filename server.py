import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Set Streamlit page config
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

st.title("📊 Student Performance Analysis Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader("student_data_full.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Raw Data")
    st.dataframe(df)

    # Convert categorical columns to numerical
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include="object").columns:
        df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))

    # --- Exploratory Analysis ---
    st.subheader("📈 Data Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### CGPA Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["CGPA"], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("#### Attendance vs CGPA")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x="Attendance", y="CGPA", hue="Year", ax=ax2)
        st.pyplot(fig2)

    # --- ML Prediction ---
    st.subheader("🤖 Predict CGPA using Student Features")

    feature_cols = ["Attendance", "IA marks", "Year"]  # Adjust based on your CSV
    if not all(col in df_clean.columns for col in feature_cols + ["CGPA"]):
        st.warning("Required columns for prediction not found.")
    else:
        X = df_clean[feature_cols]
        y = df_clean["CGPA"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        st.write("**Model Evaluation:**")
        st.write(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
        st.write(f"R² Score: {r2_score(y_test, predictions):.2f}")

        st.write("### Try predicting CGPA for a custom student:")
        att = st.slider("Attendance (%)", 0, 100, 75)
        ia = st.slider("IA Marks", 0, 50, 25)
        year = st.selectbox("Year", sorted(df["Year"].unique()))
        year_enc = LabelEncoder().fit(df["Year"]).transform([year])[0]

        input_data = np.array([[att, ia, year_enc]])
        cgpa_pred = model.predict(input_data)[0]
        st.success(f"📌 Predicted CGPA: {cgpa_pred:.2f}")
