import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, confusion_matrix

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("🎓 Student Performance Analysis Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your student dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Raw Data")
    st.dataframe(df)

    # Prepare data
    df_clean = df.copy()
    label_encoders = {}
    for col in df_clean.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le

    # Add category based on CGPA
    def categorize_cgpa(cgpa):
        if cgpa >= 8:
            return "Good"
        elif cgpa >= 6:
            return "Average"
        else:
            return "Poor"

    df['Performance'] = df['CGPA'].apply(categorize_cgpa)
    df_clean['Performance'] = LabelEncoder().fit_transform(df['Performance'])

    # Visualizations (no seaborn)
    st.subheader("📈 Visualizations")

    fig1, ax1 = plt.subplots()
    ax1.hist(df["CGPA"], bins=10, color='skyblue', edgecolor='black')
    ax1.set_title("CGPA Distribution")
    ax1.set_xlabel("CGPA")
    ax1.set_ylabel("Number of Students")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.scatter(df["Attendance"], df["CGPA"], color='orange')
    ax2.set_title("Attendance vs CGPA")
    ax2.set_xlabel("Attendance (%)")
    ax2.set_ylabel("CGPA")
    st.pyplot(fig2)

    # Predict CGPA (Regression)
    st.subheader("🤖 Predict CGPA")

    regression_features = ["Attendance", "IA marks", "Year"]
    if all(col in df_clean.columns for col in regression_features + ["CGPA"]):
        X_reg = df_clean[regression_features]
        y_reg = df_clean["CGPA"]

        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        reg_model = RandomForestRegressor()
        reg_model.fit(X_train_reg, y_train_reg)
        y_pred_reg = reg_model.predict(X_test_reg)

        st.markdown(f"**MAE:** {mean_absolute_error(y_test_reg, y_pred_reg):.2f}")
        st.markdown(f"**R² Score:** {r2_score(y_test_reg, y_pred_reg):.2f}")

        st.markdown("### Predict CGPA for a Custom Student")
        att = st.slider("Attendance (%)", 0, 100, 75)
        ia = st.slider("IA Marks", 0, 50, 25)
        year = st.selectbox("Year", sorted(df["Year"].unique()))
        year_enc = label_encoders["Year"].transform([year])[0]

        custom_input = np.array([[att, ia, year_enc]])
        predicted_cgpa = reg_model.predict(custom_input)[0]
        st.success(f"📌 Predicted CGPA: {predicted_cgpa:.2f}")

    # Classification (Performance Category)
    st.subheader("🎯 Predict Student Performance Category (Good / Average / Poor)")

    class_features = ["Attendance", "IA marks", "Year", "CGPA"]
    if all(col in df_clean.columns for col in class_features + ["Performance"]):
        X_cls = df_clean[class_features]
        y_cls = df_clean["Performance"]

        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
        cls_model = RandomForestClassifier()
        cls_model.fit(X_train_cls, y_train_cls)
        y_pred_cls = cls_model.predict(X_test_cls)

        st.markdown("**Classification Report:**")
        st.text(classification_report(y_test_cls, y_pred_cls, target_names=["Average", "Good", "Poor"]))

        st.markdown("### Predict Category for Custom Student")
        att2 = st.slider("Attendance (%)", 0, 100, 80, key='att2')
        ia2 = st.slider("IA Marks", 0, 50, 35, key='ia2')
        year2 = st.selectbox("Year (for classifier)", sorted(df["Year"].unique()), key='year2')
        year2_enc = label_encoders["Year"].transform([year2])[0]
        cgpa2 = st.slider("CGPA", 0.0, 10.0, 7.5, step=0.1, key='cgpa2')

        input_cls = np.array([[att2, ia2, year2_enc, cgpa2]])
        pred_category = cls_model.predict(input_cls)[0]
        category_name = LabelEncoder().fit(["Good", "Average", "Poor"]).inverse_transform([pred_category])[0]
        st.success(f"🏅 Predicted Performance Category: {category_name}")

        # Feature Importance Plot
        st.markdown("### 🔍 Feature Importance (Classifier)")
        importance = cls_model.feature_importances_
        fig3, ax3 = plt.subplots()
        ax3.bar(class_features, importance, color='green')
        ax3.set_title("Feature Importance")
        ax3.set_ylabel("Importance Score")
        st.pyplot(fig3)
