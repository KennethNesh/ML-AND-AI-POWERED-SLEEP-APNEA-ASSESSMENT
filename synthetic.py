import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Set seed for reproducibility
np.random.seed(42)
n_samples = 1000

# Generate base features with some realistic assumptions:
age = np.random.randint(30, 71, size=n_samples)  # Age between 30 and 70
sex = np.random.choice([0, 1], size=n_samples)    # 0: female, 1: male

# Waist-hip ratio: depends on sex (males typically have a slightly higher ratio)
waist_hip_ratio = np.where(sex == 1,
                           np.random.normal(0.95, 0.05, n_samples),
                           np.random.normal(0.85, 0.05, n_samples))

active_smoking = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
passive_smoking = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
alcohol = np.random.randint(0, 15, size=n_samples)           # 0 to 14 drinks per week
physical_activity = np.random.randint(0, 11, size=n_samples)   # Score 0 (sedentary) to 10 (active)
diet_quality = np.random.randint(0, 11, size=n_samples)        # Score 0 (poor) to 10 (excellent)
mental_health = np.clip(np.random.normal(10, 3, n_samples), 0, None)  # Lower is better

# Create a DataFrame of base features.
df = pd.DataFrame({
    'age': age,
    'sex': sex,
    'waist_hip_ratio': waist_hip_ratio,
    'active_smoking': active_smoking,
    'passive_smoking': passive_smoking,
    'alcohol': alcohol,
    'physical_activity': physical_activity,
    'diet_quality': diet_quality,
    'mental_health': mental_health
})

# --- Introduce Non-Linearity ---
# We add non-linear interactions and quadratic terms manually.
# For example, we include interaction between age and waist-hip ratio,
# and a quadratic term for physical activity.
noise = np.random.normal(0, 3, n_samples)  # Moderate noise

# Define a non-linear function for the target AHI.
# The idea is that risk increases with age and waist-hip ratio,
# while being protective factors are physical activity and diet.
ahi = (0.03 * age +
       25 * waist_hip_ratio +
       5 * active_smoking +
       3 * passive_smoking +
       0.5 * alcohol -
       0.6 * physical_activity -
       0.8 * diet_quality +
       0.3 * mental_health +
       0.02 * (age * waist_hip_ratio * 100) -   # Interaction: effect scales with age & body fat distribution
       2 * (physical_activity ** 2) +          # Quadratic effect: too little or too much physical activity might be non-linear
       noise)

ahi = np.clip(ahi, 0, None)  # Ensure non-negative values

# Create a binary indicator for sleep apnea based on a threshold (e.g., AHI > 15)
sleep_apnea = (ahi > 15).astype(int)

# Add the target variables to the DataFrame.
df['ahi'] = ahi
df['sleep_apnea'] = sleep_apnea

# Save the improved synthetic dataset as a CSV file.
df.to_csv('improved_synthetic_sleep_apnea_data.csv', index=False)
print("Improved synthetic dataset saved as 'improved_synthetic_sleep_apnea_data.csv'")
 