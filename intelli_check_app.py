import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
import pickle

# ---------------------------
# ⚙️ Module 1: Battery Health Predictor
# ---------------------------
np.random.seed(42)
battery_data = pd.DataFrame({
    'cycles': np.random.randint(100, 1000, 500),
    'temperature_C': np.random.normal(35, 5, 500),
    'voltage_V': np.random.normal(3.7, 0.2, 500),
    'current_A': np.random.normal(0.5, 0.1, 500),
    'usage_hours': np.random.randint(200, 5000, 500)
})

battery_data['health_percent'] = 100 - (
    battery_data['cycles'] * 0.05 + battery_data['temperature_C'] * 0.3 + battery_data['usage_hours'] * 0.01)
battery_data['health_percent'] = battery_data['health_percent'].clip(lower=10)

X = battery_data.drop('health_percent', axis=1)
y = battery_data['health_percent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

battery_model = LinearRegression()
battery_model.fit(X_train, y_train)

simulated_X = X_train.sample(10, random_state=1)
simulated_preds = battery_model.predict(simulated_X)

manual_inputs = pd.DataFrame([
    {'cycles': 600, 'temperature_C': 40, 'voltage_V': 3.6, 'current_A': 0.52, 'usage_hours': 2200},
    {'cycles': 300, 'temperature_C': 35, 'voltage_V': 3.7, 'current_A': 0.48, 'usage_hours': 1100}
])
manual_preds = battery_model.predict(manual_inputs)

all_preds = np.concatenate([simulated_preds, manual_preds])
labels = [f'Simulated {i+1}' for i in range(10)] + ['Manual 1', 'Manual 2']
colors = ['skyblue'] * 10 + ['orange'] * 2

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, all_preds, color=colors)
for i in range(len(bars) - 2, len(bars)):
    bars[i].set_edgecolor('red')
    bars[i].set_linewidth(2)

plt.xticks(rotation=45)
plt.ylabel('Predicted Health (%)')
plt.title('Battery Health Predictions – 10 Simulated + 2 Manual Readings')
plt.ylim(0, 110)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---------------------------
# ⚡ Module 2: Solder Fault Detector
# ---------------------------
np.random.seed(101)
fault_data = pd.DataFrame({
    'voltage_V': np.round(np.random.normal(5, 0.8, 300), 2),
    'current_A': np.round(np.random.normal(0.5, 0.2, 300), 2),
    'resistance_Ohm': np.round(np.random.normal(10, 2, 300), 2),
    'temperature_C': np.round(np.random.normal(38, 5, 300), 1)
})

fault_data['fault_flag'] = np.where(
    (fault_data['voltage_V'] > 6.0) |
    (fault_data['current_A'] > 0.9) |
    (fault_data['temperature_C'] > 45),
    1, 0
)

X2 = fault_data.drop('fault_flag', axis=1)
y2 = fault_data['fault_flag']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

clf_fault = RandomForestClassifier(n_estimators=100, random_state=42)
clf_fault.fit(X_train2, y_train2)

# ---------------------------
# 🧠 Module 3: Component Status Classifier
# ---------------------------
np.random.seed(303)
status_data = pd.DataFrame({
    'voltage_V': np.round(np.random.normal(5.0, 0.5, 300), 2),
    'current_A': np.round(np.random.normal(0.5, 0.1, 300), 2),
    'resistance_Ohm': np.round(np.random.normal(10, 2, 300), 2),
    'temperature_C': np.round(np.random.normal(37, 4.5, 300), 1)
})

def classify_component(row):
    if row['temperature_C'] > 45 or row['current_A'] > 0.9:
        return 2  # Faulty
    elif row['voltage_V'] < 4.5 or row['temperature_C'] > 42:
        return 1  # Aging
    else:
        return 0  # Healthy

status_data['component_status'] = status_data.apply(classify_component, axis=1)
X3 = status_data.drop('component_status', axis=1)
y3 = status_data['component_status']
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=42)

clf_status = RandomForestClassifier(n_estimators=100, random_state=42)
clf_status.fit(X_train3, y_train3)

# ---------------------------
# 🌐 Streamlit Interface
# ---------------------------
st.title("🧠 IntelliCheck – Component Status Predictor")

module = st.selectbox("Select Diagnostic Module:", ["Battery Health Predictor","Component Status Classifier", "Fault Detector"])
if module == "Battery Health Predictor":
    st.subheader("🔋 Enter Battery Parameters")

    cycles = st.number_input("Charge Cycles", 0, 2000, 500)
    temperature = st.number_input("Temperature (°C)", 0.0, 100.0, 35.0)
    voltage = st.number_input("Voltage (V)", 0.0, 5.0, 3.7)
    current = st.number_input("Current (A)", 0.0, 2.0, 0.5)
    usage = st.number_input("Usage Hours", 0, 10000, 2000)

    if st.button("Predict Battery Health"):
        input_df = pd.DataFrame([{
            'cycles': cycles,
            'temperature_C': temperature,
            'voltage_V': voltage,
            'current_A': current,
            'usage_hours': usage
        }])

        prediction = battery_model.predict(input_df)[0]
        st.success(f"Predicted Battery Health: {prediction:.2f}%")
elif module == "Component Status Classifier":
    voltage = st.number_input("Voltage (V)", 0.0, 10.0, 5.0)
    current = st.number_input("Current (A)", 0.0, 2.0, 0.5)
    resistance = st.number_input("Resistance (Ω)", 0.0, 50.0, 10.0)
    temperature = st.number_input("Temperature (°C)", 0.0, 100.0, 37.0)

    if st.button("Predict Component Status"):
        input_df = pd.DataFrame([{
            'voltage_V': voltage,
            'current_A': current,
            'resistance_Ohm': resistance,
            'temperature_C': temperature
        }])
        result = clf_status.predict(input_df)[0]
        status_map = {0: "✅ Healthy", 1: "⚠️ Aging", 2: "❌ Faulty"}
        st.success(f"Component Status: {status_map[result]}")

elif module == "Fault Detector":
    voltage = st.number_input("Voltage (V)", 0.0, 10.0, 5.0, key='v')
    current = st.number_input("Current (A)", 0.0, 2.0, 0.5, key='c')
    resistance = st.number_input("Resistance (Ω)", 0.0, 50.0, 10.0, key='r')
    temperature = st.number_input("Temperature (°C)", 0.0, 100.0, 37.0, key='t')

    if st.button("Predict Fault Presence"):
        input_df = pd.DataFrame([{
            'voltage_V': voltage,
            'current_A': current,
            'resistance_Ohm': resistance,
            'temperature_C': temperature
        }])
        result = clf_fault.predict(input_df)[0]
        fault_status = "❌ Fault Detected" if result == 1 else "✅ No Fault"
        st.success(f"Solder Fault Status: {fault_status}")
