import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load preprocessed data
X = np.load('X.npy')
y = np.load('y.npy')

# Check the shapes
print(f"Shape of X (images): {X.shape}")
print(f"Shape of y (labels): {y.shape}")

if X.shape[0] != y.shape[0]:
    min_len = min(X.shape[0], y.shape[0])
    X = X[:min_len]
    y = y[:min_len]
    np.save('X.npy', X)
    np.save('y.npy', y)


# Normalize data
X = X / 255.0

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the model weights
model.save("age_detection_model.keras")
# Make predictions 
y_pred = model.predict(X_test)

# For classification purposes, we can round the predictions
y_pred = np.round(y_pred)

# Generate confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
