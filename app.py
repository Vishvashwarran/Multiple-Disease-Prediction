import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dfc1=pd.read_csv("kidney_disease - kidney_disease.csv")
dfc2=pd.read_csv("indian_liver_patient - indian_liver_patient.csv")
dfc3=pd.read_csv("parkinsons - parkinsons.csv")



st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")

# Load trained models
kidney_model = joblib.load("best_pipeline1.pkl")
liver_model = joblib.load("best_pipe2.pkl")
parkinson_model = joblib.load("best_pipe3.pkl")

# -------------------------------------------
# Sidebar Navigation WITHOUT radio buttons
# -------------------------------------------
st.sidebar.title("Navigation")

home_btn = st.sidebar.button("🏠 Home")
predict_btn = st.sidebar.button("🧪 Disease Prediction")

# Maintain navigation state
if "page" not in st.session_state:
    st.session_state.page = "Home"

if home_btn:
    st.session_state.page = "Home"

if predict_btn:
    st.session_state.page = "Predict"


# -------------------------------------------
# HOME PAGE
# -------------------------------------------
if st.session_state.page == "Home":
    st.title("🏥 Multiple Disease Prediction System")

    st.write("""
    Welcome to the AI-based *Multiple Disease Prediction App*.  
    This system helps detect:

    - 🩸 **Kidney Disease**  
    - 🤍 **Liver Disease**  
    - 🧠 **Parkinson’s Disease**  

    Click **Disease Prediction** in the sidebar to start.
    """)

    st.subheader("Heatmap for Kidney")

    fig1, ax1 = plt.subplots(figsize=(20, 12))
    sns.heatmap(dfc1.corr(numeric_only=True), annot=True,fmt=".2f", vmin=-1, vmax=1)
    st.pyplot(fig1)


    st.subheader("Heatmap for Liver")

    fig2, ax2 = plt.subplots(figsize=(20,15))
    sns.heatmap(dfc2.corr(numeric_only=True), annot=True, vmin=-1, vmax=1)
    st.pyplot(fig2)

    st.subheader("Heatmap for Parkinson")

    fig3, ax3 = plt.subplots(figsize=(20,15))
    sns.heatmap(dfc3.corr(numeric_only=True), annot=True, vmin=-1, vmax=1)
    st.pyplot(fig3)

# -------------------------------------------
# PREDICTION PAGE
# -------------------------------------------
if st.session_state.page == "Predict":

    st.title("🧪 Disease Prediction")

    page = st.selectbox(
        "Select Prediction",
        ["Kidney Disease", "Liver Disease", "Parkinson’s Disease"]
    )

    # ----------------------------------------------------------
    # 1️⃣ KIDNEY DISEASE FORM
    # ----------------------------------------------------------
    if page == "Kidney Disease":
        st.title("🩸 Kidney Disease Prediction Form")

        with st.form("kidney_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age", 1, 120)
                bp = st.number_input("Blood Pressure")
                sg = st.number_input("Specific Gravity", 1.00, 1.06, step=0.01)
                al = st.number_input("Albumin", 0.0, 5.0)
                su = st.number_input("Sugar", 0.0, 5.0)

            with col2:
                rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
                pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
                pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])
                ba = st.selectbox("Bacteria", ["present", "notpresent"])
                bgr = st.number_input("Blood Glucose Random")

            with col3:
                bu = st.number_input("Blood Urea")
                sc = st.number_input("Serum Creatinine")
                sod = st.number_input("Sodium")
                pot = st.number_input("Potassium")
                hemo = st.number_input("Hemoglobin")

            pcv = st.number_input("Packed Cell Volume")
            wc = st.number_input("White Blood Cell Count")
            rc = st.number_input("Red Blood Cell Count")

            col4, col5, col6 = st.columns(3)
            with col4:
                htn = st.selectbox("Hypertension", ["yes", "no"])
                dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])

            with col5:
                cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
                appet = st.selectbox("Appetite", ["good", "poor"])

            with col6:
                pe = st.selectbox("Pedal Edema", ["yes", "no"])
                ane = st.selectbox("Anemia", ["yes", "no"])

            submit = st.form_submit_button("🔍 Predict Kidney Disease")

        if submit:
            data = pd.DataFrame([[ 
                age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc,
                sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane
            ]], columns=[
                'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc',
                'sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'
            ])

            result = kidney_model.predict(data)[0]
            prob = kidney_model.predict_proba(data)[0]

            if result == 0:
                st.success(f"✔ No Kidney Disease (Probability: {prob[0] * 100:.2f}%)")
            else:
                st.error(f"⚠ Kidney Disease Detected (Probability: {prob[1] * 100:.2f}%)")


    # ----------------------------------------------------------
    # 2️⃣ LIVER DISEASE FORM
    # ----------------------------------------------------------
    elif page == "Liver Disease":
        st.title("🤍 Liver Disease Prediction Form")

        with st.form("liver_form"):
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", 1, 100)
                gender = st.selectbox("Gender", ["Male", "Female"])
                tb = st.number_input("Total Bilirubin")
                db = st.number_input("Direct Bilirubin")
                ap = st.number_input("Alkaline Phosphotase")

            with col2:
                alat = st.number_input("Alamine Aminotransferase")
                asat = st.number_input("Aspartate Aminotransferase")
                tp = st.number_input("Total Proteins")
                alb = st.number_input("Albumin")
                agr = st.number_input("Albumin & Globulin Ratio")

            submit = st.form_submit_button("🔍 Predict Liver Disease")

        if submit:
            data = pd.DataFrame([[ 
                age, gender, tb, db, ap, alat, asat, tp, alb, agr
            ]], columns=[
                "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
                "Alkaline_Phosphotase", "Alamine_Aminotransferase",
                "Aspartate_Aminotransferase", "Total_Protiens",
                "Albumin", "Albumin_and_Globulin_Ratio"
            ])

            result = liver_model.predict(data)[0]
            prob = liver_model.predict_proba(data)[0]

            if result == 0:
                st.success(f"✔ No Liver Disease (Probability: {prob[0] * 100:.2f}%)")
            else:
                st.error(f"⚠ Liver Disease Detected (Probability: {prob[1] * 100:.2f}%)")


    # ----------------------------------------------------------
    # 3️⃣ PARKINSON DISEASE FORM
    # ----------------------------------------------------------
    else:
        st.title("🧠 Parkinson’s Disease Prediction Form")

        cols = [
            "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
            "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
            "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
            "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
            "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
        ]

        with st.form("parkinson_form"):
            col1, col2, col3, col4 = st.columns(4)

            values = []
            for i, c in enumerate(cols):
                with [col1, col2, col3, col4][i % 4]:
                    values.append(st.number_input(c, value=0.0))

            submit = st.form_submit_button("🔍 Predict Parkinson’s Disease")

        if submit:
            data = pd.DataFrame([values], columns=cols)
            result = parkinson_model.predict(data)[0]
            prob = parkinson_model.predict_proba(data)[0]

            if result == 0:
                st.success(f"✔ No Parkinson’s Disease (Probability: {prob[0] * 100:.2f}%)")
            else:
                st.error(f"⚠ Parkinson’s Disease Detected (Probability: {prob[1] * 100:.2f}%)")
