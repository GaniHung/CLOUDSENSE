import numpy as np
from tensorflow.keras.models import load_model
import argparse
import joblib
import os
from collections import Counter

def predict_maneuvers(model_path, sequences_path):
    """Loads a trained model and predicts maneuvers on new sequence data."""
    
    # --- 1. Find the corresponding encoder file ---
    encoder_path = model_path.replace('.h5', '_encoder.joblib')
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print(f"Error: Model ('{model_path}') or Encoder ('{encoder_path}') not found.")
        print("Please ensure you have trained the model first.")
        return

    # --- 2. Load the model, encoder, and new data ---
    print("Loading model and encoder...")
    model = load_model(model_path)
    encoder = joblib.load(encoder_path)
    
    print(f"Loading new sequences from '{sequences_path}'...")
    new_sequences = np.load(sequences_path)

    # --- 3. Make Predictions ---
    print("Predicting maneuvers...")
    predictions_prob = model.predict(new_sequences)
    # The output is a probability for each class, so we take the one with the highest probability
    predictions_encoded = np.argmax(predictions_prob, axis=1)

    # --- 4. Decode Predictions back to Text Labels ---
    # Convert the numeric predictions (0, 1, 2...) back to strings ('Split_S', 'Level_Turn'...)
    predicted_labels = encoder.inverse_transform(predictions_encoded)

    # --- 5. Display the Results ---
    print("\n--- Prediction Results ---")
    print(f"Total sequences analyzed: {len(predicted_labels)}")
    
    # Count the occurrences of each maneuver found
    maneuver_counts = Counter(predicted_labels)
    
    if not maneuver_counts:
        print("No maneuvers were identified.")
        return
        
    print("\nSummary of maneuvers found:")
    # Sort for consistent display
    for maneuver, count in sorted(maneuver_counts.items()):
        # We can ignore the 'No_Maneuver' label for a cleaner summary
        if maneuver != 'No_Maneuver':
            print(f"- {maneuver}: {count} sequences")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict maneuvers on new data using a trained LSTM model.")
    parser.add_argument("model_path", help="Path to the trained Keras model (.h5).")
    parser.add_argument("sequences_path", help="Path to the new, unlabeled sequences to predict on (.npy).")
    args = parser.parse_args()
    predict_maneuvers(args.model_path, args.sequences_path)