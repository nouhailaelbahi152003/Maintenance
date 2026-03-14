import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# -----------------------------
# LOGIN
# -----------------------------
USERNAME = "admin"
PASSWORD = "1234"

if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("🔐 Industrial Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Wrong username or password")
    st.stop()

# -----------------------------
# SIDEBAR MENU
# -----------------------------
st.sidebar.title("🏭 Predictive Maintenance")
menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Sensor Graphs",
        "Failure Prediction",
        "Feature Importance",
        "Model Evaluation",
        "Failure History",
        "Real Time Monitoring"
    ]
)

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("predictive_maintenance.csv")

# Nettoyer les noms de colonnes
data.columns = data.columns.str.strip()

# Features et target
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Vérifie le nom exact de la colonne target
possible_targets = ["Machine failure", "Machine_failure", "target", "Target"]
target = None
for t in possible_targets:
    if t in data.columns:
        target = t
        break

if target is None:
    st.error("❌ Target column not found in dataset. Check your CSV!")
    st.stop()

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# DASHBOARD
# -----------------------------
if menu == "Dashboard":
    st.title("🏭 Industrial Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Machines", len(data))
    col2.metric("Failures", data[target].sum())
    col3.metric("Healthy", len(data) - data[target].sum())

    st.subheader("Sensors Overview")
    st.line_chart(data[features].head(200))

# -----------------------------
# SENSOR GRAPHS
# -----------------------------
elif menu == "Sensor Graphs":
    st.title("📈 Sensor Data")
    sensor = st.selectbox("Select Sensor", features)
    fig, ax = plt.subplots()
    ax.plot(data[sensor])
    ax.set_title(sensor)
    st.pyplot(fig)

# -----------------------------
# FAILURE PREDICTION
# -----------------------------
elif menu == "Failure Prediction":
    st.title("🤖 Predict Machine Failure")
    air = st.slider("Air temperature [K]", 290, 330, 300)
    process = st.slider("Process temperature [K]", 300, 350, 310)
    speed = st.slider("Rotational speed [rpm]", 1000, 3000, 1500)
    torque = st.slider("Torque [Nm]", 0, 100, 40)
    wear = st.slider("Tool wear [min]", 0, 250, 100)

    if st.button("Predict"):
        input_data = np.array([[air, process, speed, torque, wear, 0]])
        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]
        st.write("Failure Probability:", round(prob*100,2), "%")
        if pred == 1:
            st.error("⚠️ FAILURE RISK")
            history = pd.DataFrame({
                "Air Temp":[air],
                "Process Temp":[process],
                "Speed":[speed],
                "Torque":[torque],
                "Wear":[wear]
            })
            history.to_csv("history.csv", mode="a", header=False, index=False)
        else:
            st.success("Machine Healthy")

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
elif menu == "Feature Importance":
    st.title("📊 Sensor Importance")
    importance = model.feature_importances_
    df = pd.DataFrame({
        "Sensor":features,
        "Importance":importance
    })
    st.bar_chart(df.set_index("Sensor"))

# -----------------------------
# MODEL EVALUATION
# -----------------------------
elif menu == "Model Evaluation":
    st.title("🧠 Model Performance")
    X = data[features]
    y = data[target]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    acc = accuracy_score(y, pred)
    st.write("Accuracy:", acc)
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# -----------------------------
# FAILURE HISTORY
# -----------------------------
elif menu == "Failure History":
    st.title("📜 Failure History")
    try:
        history = pd.read_csv("history.csv")
        st.dataframe(history)
    except:
        st.write("No failure recorded")

# -----------------------------
# REAL TIME MONITORING
# -----------------------------
elif menu == "Real Time Monitoring":
    st.title("📡 Real Time Sensors")
    chart = st.line_chart(pd.DataFrame(
        np.random.randn(20,3),
        columns=["Temperature","Vibration","Load"]
    ))
    for i in range(30):
        new = pd.DataFrame(
            np.random.randn(1,3),
            columns=["Temperature","Vibration","Load"]
        )
        chart.add_rows(new)
        time.sleep(0.5)