import pandas as pd
import numpy as np
import os
import argparse

def create_sequences_from_df(df, sequence_length, feature_cols):
    """Creates sequences and labels from a single aircraft's DataFrame."""
    sequences, labels = [], []
    for col in feature_cols:
        if col not in df.columns: df[col] = 0
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    values, maneuver_labels = df[feature_cols].values, df['Maneuver_Label'].values
    for i in range(len(df) - sequence_length + 1):
        sequences.append(values[i:i+sequence_length])
        window_labels = [l for l in maneuver_labels[i:i+sequence_length] if l and l != '']
        
        if window_labels:
            label = max(set(window_labels), key=window_labels.count)
        else:
            label = 'No_Maneuver'
        labels.append(label)
        
    return np.array(sequences), np.array(labels)

def main(input_dir, output_sequences_path, output_labels_path, sequence_length):
    """Loads labeled data from a directory and prepares it for ML."""
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found '{input_dir}'.")
        print("Please ensure you have run the curation script first.")
        return

    feature_cols = ['Roll', 'Pitch', 'Yaw', 'Speed_ms', 'Altitude', 'VS_ms', 'G_Normal', 'G_Axial', 'G_Lateral', 'RollRate', 'PitchRate', 'YawRate', 'TurnRate', 'SpecificEnergy', 'SpecificPower']
    all_sequences, all_labels = [], []
    print(f"Loading and creating sequences from files in '{input_dir}'...")

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_dir, filename))
            if len(df) >= sequence_length:
                sequences, labels = create_sequences_from_df(df, sequence_length, feature_cols)
                if len(sequences) > 0:
                    all_sequences.append(sequences); all_labels.append(labels)
            
    if not all_sequences:
        print("No sequences were created. Check data length and sequence length."); return
        
    final_sequences, final_labels = np.concatenate(all_sequences), np.concatenate(all_labels)
    os.makedirs(os.path.dirname(output_sequences_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)
    np.save(output_sequences_path, final_sequences)
    np.save(output_labels_path, final_labels)
    print(f"\nML data preparation complete. Shapes: {final_sequences.shape}, {final_labels.shape}")
    print(f"Data saved to '{output_sequences_path}' and '{output_labels_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare labeled flight data for ML training.")
    parser.add_argument("input_dir", help="Directory containing the curated data folders (e.g., '..._Curated_For_ML/').")
    parser.add_argument("output_sequences", help="Path to save the output sequences (.npy).")
    parser.add_argument("output_labels", help="Path to save the output labels (.npy).")
    parser.add_argument("--sequence_length", type=int, default=20, help="The number of time steps for each sequence.")
    args = parser.parse_args()
    main(args.input_dir, args.output_sequences, args.output_labels, args.sequence_length)