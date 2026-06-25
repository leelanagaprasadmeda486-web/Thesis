import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

print("🤖 Step 1: Loading structured NASA telemetry data...")
df = pd.read_csv("nasa_b0005_clean.csv")

# Train our baseline model on Cycle 1 (Our pristine, healthy baseline reference)
healthy_cycle_data = df[df['Cycle_Index'] == 1]
X_train = healthy_cycle_data[['Time', 'Temperature']]
y_train = healthy_cycle_data['Voltage']

print("⚙️ Step 2: Training Advanced ML Regressor (Gradient Boosting)...")
ml_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
ml_model.fit(X_train, y_train)

# -------------------------------------------------------------
# STEP 3: SIMULATE AN INCOMING LOG WITH A DEVELOPING FAULT
# -------------------------------------------------------------
print("⚠️ Step 3: Simulating a live cell run with an Internal Short Circuit...")
test_cycle_data = df[df['Cycle_Index'] == 2].copy()

# Inject an Internal Short Circuit (Voltage Drop) starting at Time step = 1500 seconds
fault_start_ground_truth = 1500
test_cycle_data.loc[test_cycle_data['Time'] >= fault_start_ground_truth, 'Voltage'] -= 0.18

# -------------------------------------------------------------
# STEP 4: GENERATE ERROR RESIDUAL SIGNALS
# -------------------------------------------------------------
X_test = test_cycle_data[['Time', 'Temperature']]
predicted_healthy_voltage = ml_model.predict(X_test)
residuals = predicted_healthy_voltage - test_cycle_data['Voltage']

# -------------------------------------------------------------
# STEP 5: DYNAMIC PROGRAMMING SEQUENTIAL FAULT TRACKER
# -------------------------------------------------------------
print("🔍 Step 4: Running Dynamic Programming Fault Isolation Loop...")
n_steps = len(residuals)
dp_score = np.zeros(n_steps)
anomaly_threshold = 0.05  # Allowable sensor dead-band limit
fault_flagged = False
detected_fault_time = -1

time_array = test_cycle_data['Time'].values

for t in range(1, n_steps):
    step_cost = residuals.iloc[t] - anomaly_threshold
    dp_score[t] = max(0, dp_score[t-1] + step_cost)
    
    if dp_score[t] > 1.0 and not fault_flagged:
        fault_flagged = True
        # Pull only the first element [0] where the cumulative curve broke away from zero
        zero_indices = np.where(dp_score[:t] == 0)[0]
        if len(zero_indices) > 0:
            start_idx = zero_indices[-1] + 1
            detected_fault_time = time_array[start_idx]
        else:
            detected_fault_time = time_array[0]

print(f"\n📋 [NASA ADVANCED DIAGNOSTIC REPORT] 📋")
print(f"Cell Health Status Anomalous: {fault_flagged}")
if fault_flagged:
    print(f"Exact Time of Fault Inception Pinpointed by DP: {detected_fault_time} seconds.")
    print(f"Ground Truth Injected Fault Timeline: {fault_start_ground_truth} seconds.")

# -------------------------------------------------------------
# STEP 6: GENERATE THE PUBLICATION VISUAL ASSETS
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(time_array, test_cycle_data['Voltage'], label='Actual Cell Telemetry (Fault Injected)', color='black')
plt.plot(time_array, predicted_healthy_voltage, label='ML Expected Healthy Baseline', color='green', linestyle='--')
plt.ylabel('Terminal Voltage (V)')
plt.title('IEEE Thesis Asset: ML-DP Integrated Battery Diagnostic Architecture')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_array, dp_score, label='DP Cumulative Fault Evidence Score', color='red', linewidth=2)
plt.axvline(x=fault_start_ground_truth, color='gray', linestyle=':', label='True Fault Inception (1500s)')
if fault_flagged:
    plt.axvline(x=float(detected_fault_time), color='magenta', linestyle='--', label=f'DP Isolated Time ({detected_fault_time}s)')
plt.xlabel('Time (Seconds)')
plt.ylabel('DP Cumulative Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('nasa_fault_isolation_result.png')
print("\n📈 Publication-ready asset saved as 'nasa_fault_isolation_result.png'")
plt.show()
