import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
import os
import argparse
import joblib

def train_lstm(sequences_path, labels_path, model_path):
    """Trains an LSTM model and saves both the model and its label encoder."""
    if not os.path.exists(sequences_path) or not os.path.exists(labels_path):
        print("Error: Input sequence or label file not found.")
        return

    sequences = np.load(sequences_path)
    labels = np.load(labels_path)

    unique_labels = np.unique(labels)
    print(f"Found {len(unique_labels)} unique labels: {unique_labels}")

    if len(unique_labels) <= 1:
        print("Error: Cannot train model with only one class.")
        return

    # --- This is the encoder we need to save ---
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels_categorical, test_size=0.2, random_state=42, stratify=labels_categorical
    )

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(y_train.shape[1], activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    print("Starting model training...")
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=2)

    # --- Save the Model and the Encoder ---
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        
    # 1. Save the Keras model
    model.save(model_path)
    print(f"Trained model saved to {model_path}")

    # 2. Save the LabelEncoder next to the model
    encoder_path = model_path.replace('.h5', '_encoder.joblib')
    joblib.dump(encoder, encoder_path)
    print(f"Label encoder saved to {encoder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM model.")
    parser.add_argument("sequences_path", help="Path to the input sequences (.npy).")
    parser.add_argument("labels_path", help="Path to the input labels (.npy).")
    parser.add_argument("model_path", help="Path to save the trained model (.h5).")
    args = parser.parse_args()
    train_lstm(args.sequences_path, args.labels_path, args.model_path)