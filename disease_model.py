import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
import joblib
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np

# Load dataset
df = pd.read_csv("merged_dataset.csv")
df.columns = df.columns.str.lower().str.strip()

# Define label columns
health_flags = ['diabetes', 'ht', 'asthma', 'fever', 'smoker', 'cld', 'ihd']

# Ensure label columns are binary integers
for col in health_flags:
    df[col] = df[col].fillna(0).astype(int)

# Print label distribution
print("\nLabel distribution:")
print(df[health_flags].sum())

# Extract feature columns (0 to 31 as strings)
feature_cols = [str(i) for i in range(32)]

# Prepare feature matrix and label matrix
X = df[feature_cols]
y = df[health_flags]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to numpy arrays for iterative splitting
X_np = np.array(X_scaled)
y_np = np.array(y)

# Perform iterative stratified split
X_train, y_train, X_test, y_test = iterative_train_test_split(X_np, y_np, test_size=0.2)

# Define multilabel classification model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(health_flags), activation='sigmoid')  
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, validation_split=0.1, epochs=30, batch_size=16, verbose=1)

# Predict and binarize
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary, target_names=health_flags))

# Save model and scaler
model.save("health_condition_multilabel_model.h5")
joblib.dump(scaler, "input_scaler.pkl")

print("✅ Model saved as 'health_condition_multilabel_model.h5'")
print("✅ Scaler saved as 'input_scaler.pkl'")
