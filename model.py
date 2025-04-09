import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

# Load synthetic data from the CSV file.
data = pd.read_csv('synthetic_sleep_apnea_data.csv')

# Define input features and target.
features = ['age', 'sex', 'waist_hip_ratio', 'active_smoking', 
            'passive_smoking', 'alcohol', 'physical_activity', 
            'diet_quality', 'mental_health']
X = data[features].values
y = data['sleep_apnea'].values  # Binary target (0 or 1)

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build a deeper neural network with additional layers and Batch Normalization.
model = Sequential([
    Dense(32, input_dim=len(features), activation='relu'),
    BatchNormalization(),
    Dropout(0.2),  # Dropout to reduce overfitting
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dense(8, activation='relu'),
    BatchNormalization(),
    Dense(4, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model.
model.compile(loss='binary_crossentropy', 
              optimizer=Adam(learning_rate=0.01), 
              metrics=['accuracy'])

# Train the model.
model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=1)

# Evaluate the model.
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.2f}")
# After training your model, save it as an HDF5 file
model.save('sleep_apnea_model1.h5')
print("Trained model saved as 'sleep_apnea_model.h5'")
