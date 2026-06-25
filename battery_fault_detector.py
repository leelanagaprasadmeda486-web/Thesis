import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

print("🤖 Loading data and retraining our baseline healthy model...")
df = pd.read_csv("us06_drive_cycle.csv")
X = df[['Current_Demand', 'Temperature']]
y = df['Voltage']

# Train on the whole clean dataset to lock in "Healthy Behavior"
ml_model = RandomForestRegressor(n_estimators=50, random_state=42)
ml_model.fit(X, y)

# -------------------------------------------------------------
# STEP 1: SIMULATE A REAL-WORLD RUN WITH AN INJECTED FAULT
# -------------------------------------------------------------
print("⚠️ Simulating an incoming battery data stream...")
df_test = df.copy()

# Inject an artificial Internal Short Circuit (Voltage Drop) starting at Time step 700
df_test.loc[700:, 'Voltage'] = df_test.loc[700:, 'Voltage'] - 0.25 

# Add random real-world sensor noise to make it realistic
np.random.seed(10)
df_test['Voltage'] = df_test['Voltage'] + np.random.normal(0, 0.015, len(df_test))

# -------------------------------------------------------------
# STEP 2: GENERATE RESIDUALS USING OUR ML MODEL
# -------------------------------------------------------------
predicted_healthy_voltage = ml_model.predict(df_test[['Current_Demand', 'Temperature']])
# Residual = Predicted Healthy behavior - Actual Incoming Behavior
residuals = predicted_healthy_voltage - df_test['Voltage']

# -------------------------------------------------------------
# STEP 3: DYNAMIC PROGRAMMING SEQUENTIAL DETECTOR
# -------------------------------------------------------------
print("⚙️ Running Dynamic Programming Fault Isolation...")
n_steps = len(residuals)
dp_score = np.zeros(n_steps)
anomaly_threshold = 0.08  # Tolerable deviation limit
fault_detected = False
fault_time = -1

# DP Recurrence loop tracking sequential deviations over time
for t in range(1, n_steps):
    step_cost = residuals[t] - anomaly_threshold
    # Bellman optimization step: maximize evidence accumulation or reset to 0
    dp_score[t] = max(0, dp_score[t-1] + step_cost)
    
    # Check if cumulative anomaly evidence breaches our confidence limit
    if dp_score[t] > 1.5 and not fault_detected:
        fault_detected = True
        # Trace back to find where this accumulation path initiated
        fault_time = np.where(dp_score[:t] == 0)[0][-1] + 1

print(f"\n🚨 [DP DIAGNOSTIC REPORT] 🚨")
print(f"Fault Status Flagged: {fault_detected}")
if fault_detected:
    print(f"Exact Time Index of Fault Inception Pinpointed by DP: {fault_time} seconds.")
    print(f"Ground Truth Injected Fault Time: 700 seconds.")

# -------------------------------------------------------------
# STEP 4: PLOT THE FINAL THESIS-READY OUTPUT
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(df_test['Time'], df_test['Voltage'], label='Actual Telemetry (Faulty after 700s)', color='black')
plt.plot(df_test['Time'], predicted_healthy_voltage, label='ML Expected Healthy Trend', color='green', linestyle='--')
plt.ylabel('Voltage (V)')
plt.legend()
plt.title('ML-DP Battery Fault Detection Pipeline')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(df_test['Time'], dp_score, label='DP Cumulative Fault Evidence Score', color='red', linewidth=2)
plt.axvline(x=700, color='gray', linestyle=':', label='True Fault Injection (700s)')
if fault_detected:
    plt.axvline(x=fault_time, color='magenta', linestyle='--', label=f'DP Detected Time ({fault_time}s)')
plt.xlabel('Time (Seconds)')
plt.ylabel('DP Cumulative Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('battery_fault_detection_result.png')
print("\n📈 Final verification plot saved as 'battery_fault_detection_result.png'")
plt.show()
