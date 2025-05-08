import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, classification_report

# Page setup
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.markdown("""
<h1 style='text-align: center; color: #4CAF50; font-size: 48px;'>
    🎓 Student Performance Analysis Dashboard 
</h1>
<hr style='border-top: 3px solid #4CAF50;'>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("student_data_full.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Raw Data")
    st.dataframe(df)

    # Label Encoding for ML
    df_clean = df.copy()
    label_encoders = {}
    for col in df_clean.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le

    # Performance category
    def categorize_cgpa(cgpa):
        if cgpa >= 8:
            return "Good"
        elif cgpa >= 6:
            return "Average"
        else:
            return "Poor"
    df['Performance'] = df['CGPA'].apply(categorize_cgpa)
    df_clean['Performance'] = LabelEncoder().fit_transform(df['Performance'])

    # Job Suitability
    def job_prediction(row):
        if row["CGPA"] < 7 and row["Attendance"] < 75:
            return "❌ Unfit for Job"
        skills = str(row["skills"]).lower() + " " + str(row["interest"]).lower()
        if any(word in skills for word in ["dev", "software", "coding", "programming"]):
            return "👨‍💻 Software Engineering"
        if any(word in skills for word in ["test", "qa", "automation", "selenium"]):
            return "🧪 Automation Testing"
        return "🔍 Undecided"
    df["Job Suitability"] = df.apply(job_prediction, axis=1)

    # Placement Offer
    def placement_prediction(row):
        skills = str(row["skills"]).lower() + " " + str(row["interest"]).lower()
        keywords = ["dev", "python", "java", "testing", "automation", "sql", "c++", "web"]
        has_skills = any(word in skills for word in keywords)
        if row["CGPA"] >= 7.5 and row["Attendance"] >= 80 and has_skills:
            return "✅ Yes"
        return "❌ No"
    df["Placement Offer"] = df.apply(placement_prediction, axis=1)

    # 📈 Visualizations
    st.subheader("📊 Data Visualizations")

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

    # 🔮 CGPA Regression
    st.subheader("🤖 Predict CGPA")

    regression_features = ["Attendance", "IA marks", "Year"]
    if all(col in df_clean.columns for col in regression_features + ["CGPA"]):
        X_reg = df_clean[regression_features]
        y_reg = df_clean["CGPA"]
        X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

        reg_model = RandomForestRegressor()
        reg_model.fit(X_train, y_train)
        y_pred = reg_model.predict(X_test)

        st.markdown(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
        st.markdown(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

        st.markdown("### Predict CGPA for a Custom Student")
        att = st.slider("Attendance (%)", 0, 100, 75)
        ia = st.slider("IA Marks", 0, 50, 25)
        year = st.selectbox("Year", sorted(df["Year"].unique()))
        year_enc = label_encoders["Year"].transform([year])[0]

        custom_input = np.array([[att, ia, year_enc]])
        pred_cgpa = reg_model.predict(custom_input)[0]
        st.success(f"📌 Predicted CGPA: {pred_cgpa:.2f}")

    # 🎯 Classification
    st.subheader("🎯 Predict Performance Category (Good / Average / Poor)")

    class_features = ["Attendance", "IA marks", "Year", "CGPA"]
    if all(col in df_clean.columns for col in class_features + ["Performance"]):
        X_cls = df_clean[class_features]
        y_cls = df_clean["Performance"]
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

        cls_model = RandomForestClassifier()
        cls_model.fit(X_train_c, y_train_c)
        y_pred_c = cls_model.predict(X_test_c)

        st.text(classification_report(y_test_c, y_pred_c, target_names=["Average", "Good", "Poor"]))

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

    # 🧠 Job + Placement Results
    st.subheader("🧠 Final Predictions (Job & Placement)")

    st.dataframe(df[["name", "CGPA", "Attendance", "skills", "interest", "Job Suitability", "Placement Offer"]])
